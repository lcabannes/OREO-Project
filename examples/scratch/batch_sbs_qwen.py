from typing import List, Optional
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
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math


def expand(a: LLM, states: List[List[str]], which_data) -> List[List[dict]]:
    llm_inputs = []
    for i in range(len(states)):
        for state in states[i]:
            llm_inputs.extend([state] * args.K)
    if which_data == "GSM8K":
        params = SamplingParams(max_tokens=2048, stop="\n", temperature=1)
    else:
        params = SamplingParams(max_tokens=2048, stop=["\n", ". ", ".$ "], temperature=1)
    results = a.generate(llm_inputs, params)
    next_states: List[List[dict]] = []
    j = 0
    for i in range(len(states)):
        next_states.append([])
        for state in states[i]:
            for _ in range(args.K):
                next_states[-1].append(
                    {
                        "text": state + results[j].outputs[0].text,
                        "stop_reason": results[j].outputs[0].stop_reason,
                    }
                )
                j += 1
    return next_states


def compute_values(tokenizer: PreTrainedTokenizer, critic, next_states: List[List[dict]]):
    bsz = 8

    str_batch = []
    values_buffer = []
    for i in tqdm(range(len(next_states)), desc="Compute Values"):
        for next_state in next_states[i]:
            if args.add_eos:
                str_batch.append(next_state["text"] + " " + tokenizer.eos_token)
            else:
                str_batch.append(next_state["text"])
            if len(str_batch) == bsz:
                input_token = tokenizer(str_batch, padding=True, return_tensors="pt")
                input_token.to(torch.cuda.current_device())
                with torch.no_grad():
                    values: torch.Tensor = critic(**input_token, action_mask=input_token["input_ids"])
                    values_buffer.extend(values[:, -1].tolist())
                str_batch = []
    if len(str_batch) != 0:
        input_token = tokenizer(str_batch, padding=True, return_tensors="pt")
        input_token.to(torch.cuda.current_device())
        with torch.no_grad():
            values: torch.Tensor = critic(**input_token, action_mask=input_token["input_ids"])
            values_buffer.extend(values[:, -1].tolist())

    results: List[List[float]] = []
    j = 0
    for i in range(len(next_states)):
        results.append([])
        for next_state in next_states[i]:
            results[-1].append(values_buffer[j])
            j += 1
    return results


def select_states(
    next_states: List[List[dict]], values: List[List[float]], answers: Optional[List[List[dict]]] = None
):
    states: List[List[str]] = []
    flag = False
    for i in range(len(next_states)):
        candidates = []
        assert len(next_states[i]) == len(values[i])
        for next_state, value in zip(next_states[i], values[i]):
            if next_state["stop_reason"] is None:
                if answers is not None:
                    answers[i].append(next_state)
                else:
                    candidates.append((next_state["text"], value))
            else:
                candidates.append((next_state["text"] + str(next_state["stop_reason"]), value))

        if len(candidates) > 0:
            flag = True

        candidates = sorted(candidates, key=lambda x: -x[1])
        states.append([candidate[0] for candidate in candidates[: args.K]])
    return states, flag


def beam_search(a: LLM, tokenizer: PreTrainedTokenizer, critic, prompts: List[str], which_data) -> List[str]:
    states = [[x] for x in prompts]
    answers = [[] for _ in range(len(prompts))]

    for _ in tqdm(range(20), desc="Step"):
        next_states = expand(a, states, which_data)
        values = compute_values(tokenizer, critic, next_states)
        states, flag = select_states(next_states, values, answers)
        if not flag:
            break

        # for i in range(len(states)):
        #     for k in range(len(states[i])):
        #         states[i][k] += "\n"

    values = compute_values(tokenizer, critic, answers)
    states, _ = select_states(answers, values)
    return [state[0] if len(state) > 0 else None for state in states]


def main():
    if args.data == "GSM8K":
        print(f"chose GSM8K")
        d = load_dataset("openai/gsm8k", "main")[args.split].select(range(300))
        extract_answer = extract_last_single_answer
        eval_answer = eval_last_single_answer
        input_key = "question"
        output_key = "answer"
    else:
        print(f"chose MATH")
        d = load_dataset("hendrycks/competition_math")[args.split].select(range(300))
        extract_answer = extract_math_answer
        eval_answer = eval_math
        input_key = "problem"
        output_key = "solution"

    model_path = "/home/OREO-Project/data/pretrained/Qwen2.5-Math-1.5B-OREO/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    critic = get_llm_for_sequence_regression(
        model_path + "_critic", # "Qwen/Qwen2.5-Math-1.5B-Instruct", # "data/pretrained/Qwen2.5-Math-1.5B-OREO/_critic/",  # 
        "critic",
        normalize_reward=False,  # TODO: maybe experiment with this layer
        use_flash_attention_2=True,
        bf16=True,
        load_in_4bit=False,
        lora_rank=64, # 64,
        lora_alpha=64,
        lora_dropout=0,
        target_modules="all-linear",
        zero_init_value_head=True,
        # ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    critic.to(torch.cuda.current_device())
    # critic.load_adapter(model_path + "_lora_critic", "default")
    critic.eval()

    model_path = "data/ckpt/Qwen-math-oreo/_merged/"
    model_path = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    a = LLM(model_path, gpu_memory_utilization=0.5)

    questions = [d[i][input_key] for i in range(len(d))]
    # prompts = [chat_template.format(x + template) for x in questions]
    prompts = tokenizer.apply_chat_template(
        [
            [
                {
                    "role": "system",
                    "content": (
                        "Please reason step by step, and produce a final answer following 4 '#', like '#### 0'"
                        if args.data == "GSM8K"
                        else "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                },
                {"role": "user", "content": x},
            ]
            for x in questions
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    completions = beam_search(a, tokenizer, critic, prompts, args.data)
    assert len(questions) == len(completions)

    num_corrects = 0
    with open(args.save, "w") as f:
        for i in tqdm(range(len(questions)), desc="Eval"):
            if completions[i] is None:
                f.write(
                    json.dumps(
                        {
                            "prompt": prompts[i],
                            "question": questions[i],
                            "answer": "",
                            "correct": False,
                        }
                    )
                )
                f.write("\n")
                continue
            assert completions[i].startswith(prompts[i])
            resp = completions[i][len(prompts[i]) :]
            pred = extract_answer(questions[i], resp, "")
            answer = extract_answer(questions[i], d[i][output_key], "")
            correct = eval_answer({"prediction": pred, "answer": answer})
            num_corrects += correct
            f.write(
                json.dumps(
                    {
                        "prompt": prompts[i],
                        "question": questions[i],
                        "answer": resp,
                        "correct": correct,
                    }
                )
            )
            f.write("\n")
    print("{} / {}".format(num_corrects, len(questions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["GSM8K", "MATH"])
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--add_eos", action="store_true")
    args = parser.parse_args()

    main()
