from typing import List
import argparse
import json

from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"


def generate(
    a: LLM, N: int, prompts: List[str], questions: List[str], answers: List[str], extract_answer, eval_answer
):
    params = SamplingParams(temperature=args.temperature, max_tokens=2048 if not args.metamath else 1024)

    completions: List[List[str]] = [[] for _ in range(len(prompts))]
    for _ in range(N):
        results = a.generate(prompts, params)
        for i in range(len(results)):
            completions[i].append(results[i].outputs[0].text)

    corrects: List[List[bool]] = [[] for _ in range(len(prompts))]
    for i, item in enumerate(completions):
        for completion in item:
            pred = extract_answer(questions[i], completion, "")
            answer = extract_answer(questions[i], answers[i], "")
            correct = eval_answer(
                {
                    "prediction": pred,
                    "answer": answer,
                }
            )
            corrects[i].append(correct)

    return completions, corrects


def compute_values(
    critic, tokenizer: PreTrainedTokenizer, prompts: List[str], completions: List[List[str]], bsz: int = 16
):
    str_batch = []
    results_flat = []
    for i, prompt in enumerate(tqdm(prompts)):
        for j, completion in enumerate(completions[i]):
            str_batch.append(prompt + completion)
            if len(str_batch) == bsz or (i == len(prompts) - 1 and j == len(completions[i]) - 1):
                input_token = tokenizer(str_batch, padding=True, return_tensors="pt")
                input_token.to(torch.cuda.current_device())
                with torch.no_grad():
                    values = critic(**input_token, action_mask=input_token["input_ids"])  # [bsz, seq_len]
                for k in range(values.shape[0]):
                    results_flat.append(
                        {
                            "last": values[k, -1].item(),
                            "second_last": values[k, -2].item(),
                            "min": values[k].min().item(),
                            "mean": values[k].mean().item(),
                        }
                    )
                str_batch = []
    k = 0
    results: List[List[dict]] = [[] for _ in prompts]
    for i, prompt in enumerate(prompts):
        for j, completion in enumerate(completions[i]):
            results[i].append(results_flat[k])
            k += 1
    return results


def plot(corrects: List[List[bool]], values: List[List[dict]]):
    N = len(corrects[0])
    results = np.zeros((len(corrects), N))
    for i in range(len(corrects)):
        assert len(corrects[i]) == N and len(values[i]) == N
        for j in range(N):
            k = np.argmax([item["last"] for item in values[i][: j + 1]])
            results[i, j] = corrects[i][k]

    acc = results.mean(0)
    print(acc[-1])
    plt.plot(range(1, N + 1), acc)
    plt.savefig("tmp-bon-full-gsm8k-plot.jpg")


def main():
    if args.data == "GSM8K":
        d = load_dataset("openai/gsm8k", "main")[args.split]
        extract_answer = extract_last_single_answer
        eval_answer = eval_last_single_answer
        input_key = "question"
        output_key = "answer"
    else:
        d = load_dataset("hendrycks/competition_math")[args.split]
        extract_answer = extract_math_answer
        eval_answer = eval_math
        input_key = "problem"
        output_key = "solution"

    questions = [x[input_key] for x in d]
    answers = [x[output_key] for x in d]
    prompts = [chat_template.format(question + template) for question in questions]

    if args.load_generation is not None:
        completions = []
        corrects = []
        with open(args.load_generation, "r") as f:
            for line in f.readlines():
                tmp = json.loads(line)
                completions.append([item["text"] for item in tmp["completions"]])
                corrects.append([item["correct"] for item in tmp["completions"]])
    if args.load_values is not None:
        values = []
        with open(args.load_values, "r") as f:
            for line in f.readlines():
                values.append(json.loads(line))

    if args.generate:
        a = LLM(args.model)

        completions, corrects = generate(a, args.N, prompts, questions, answers, extract_answer, eval_answer)

        with open(args.save, "w") as f:
            for i in range(len(completions)):
                tmp = {
                    "prompt": prompts[i],
                    "question": questions[i],
                    "answer": answers[i],
                    "completions": [
                        {"text": completions[i][j], "correct": corrects[i][j]} for j in range(len(completions[i]))
                    ],
                }
                f.write(json.dumps(tmp))
                f.write("\n")
    elif args.compute_value:
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
        critic.load_adapter(args.model + "_critic", "default")
        critic.eval()
        values = compute_values(critic, tokenizer, prompts, completions)
        with open(args.save, "w") as f:
            for item in values:
                f.write(json.dumps(item))
                f.write("\n")
    elif args.plot:
        plot(corrects, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, choices=["GSM8K", "MATH"])
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--metamath", action="store_true")

    # what to do?
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compute_value", action="store_true")
    parser.add_argument("--load_generation", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--load_values", type=str)

    parser.add_argument("--temperature", type=float, required=True)

    args = parser.parse_args()

    if args.metamath:
        template = ""
        chat_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response: Let's think step by step.\n"

    main()
