from typing import List
import json
import argparse
import itertools

from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"


def beam_search(a: LLM, tokenizer: PreTrainedTokenizer, critic, questions: List[str]) -> List[str]:
    initial_prompts = [chat_template.format(question + template) for question in questions]
    states = [[prompt] for prompt in initial_prompts]
    answers_pool = [[] for _ in range(len(questions))]

    params = SamplingParams(max_tokens=2048, stop="\n")

    for _ in range(16):
        tmp = list(itertools.chain(*states))
        if len(tmp) == 0:
            break
        results: List[List[RequestOutput]] = []
        for repeat in range(args.K):
            results.append(a.generate(tmp, params))
        offset = 0
        next_states_buffer = []
        for i in range(len(questions)):
            next_states = []
            for j in range(len(states[i])):
                for k in range(args.K):
                    next_states.append(
                        {
                            "state": states[i][j] + results[k][offset + j].outputs[0].text,
                            "stop_reason": results[k][offset + j].outputs[0].stop_reason,
                        }
                    )
                next_states_buffer.extend(next_states)
            offset += len(states[i])

        input_token = tokenizer([x["state"] for x in next_states_buffer], padding=True, return_tensors="pt")
        input_token.to(torch.cuda.current_device())
        with torch.no_grad():
            values = critic(**input_token, action_mask=input_token["input_ids"])

        offset = 0
        for i in range(len(questions)):
            next_states = next_states_buffer[offset : offset + len(states[i]) * args.K]
            for j in range(len(next_states)):
                next_states[j]["value"] = values[offset + j, -1].item()
                next_states[j]["state"] += "\n"
            next_states = sorted(next_states, key=lambda x: -x["value"])[: args.K]

            offset += len(states[i]) * args.K

            states[i] = []
            for state in next_states:
                if state["stop_reason"] is None:
                    answers_pool[i].append(state)
                else:
                    states[i].append(state["state"])

    answers = []
    for i in range(len(questions)):
        answers_pool[i] = sorted(answers_pool[i], key=lambda x: -x["value"])
        answers.append(answers_pool[i][0]["state"][len(questions[i]) :])
    return answers


def main():
    bsz = 4

    if args.data == "GSM8K":
        d = load_dataset("openai/gsm8k", "main")[args.split]
        extract_answer = extract_last_single_answer
        input_key = "question"
    else:
        d = load_dataset("hendrycks/competition_math")[args.split]
        extract_answer = extract_math_answer
        input_key = "problem"

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    critic = get_llm_for_sequence_regression(
        "deepseek-ai/deepseek-math-7b-instruct",
        "critic",
        normalize_reward=False,  # TODO: maybe experiment with this layer
        use_flash_attention_2=True,
        bf16=True,
        load_in_4bit=False,
        lora_rank=64,
        lora_alpha=64,
        lora_dropout=0,
        target_modules="all-linear",
        # ds_config=strategy.get_ds_train_config(is_actor=True),
        zero_init_value_head=True,
    )
    critic.to(torch.cuda.current_device())
    critic.load_adapter(
        "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_critic", "default"
    )
    critic.eval()
    a = LLM(
        "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_merged",
        gpu_memory_utilization=0.5,
    )

    for i in range(0, len(d), bsz):
        j = min(len(d), i + bsz)
        questions = [d[r][input_key] for r in range(i, j)]
        __import__("pdb").set_trace()
        beam_search(a, tokenizer, critic, questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["GSM8K", "MATH"])
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    main()
