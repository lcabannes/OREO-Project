import argparse
import json

import matplotlib.pyplot as plt
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_queries(dataset: Dataset, input_key: str, output_key: str, extract_func):
    result = []
    for item in dataset:
        question = item[input_key]
        response = item[output_key]
        answer = extract_func(question, response, "")
        result.append({"question": question, "answer": answer})
    return result


def gen(tokenizer, queries):
    prompts = []
    for query in queries:
        q = query["question"] + template
        q: str = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, add_special_tokens=False, tokenize=False
        )
        prompts.append(q[21:])  # TODO: any better way to remove BOS?
    return prompts


def main():
    a = LLM(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    extract_func = extract_last_single_answer if args.data == "GSM8K" else extract_math_answer
    eval_answer_func = eval_last_single_answer if args.data == "GSM8K" else eval_math

    if args.data == "GSM8K":
        d = load_dataset("openai/gsm8k", "main")[args.split]
        queries = load_queries(d, "question", "answer", extract_func)
    else:
        d = load_dataset("hendrycks/competition_math")[args.split]
        queries = load_queries(d, "problem", "solution", extract_func)
    prompts = gen(tokenizer, queries)

    indices = list(range(len(queries)))
    N, M = 1, 0
    data = [[] for _ in range(len(queries))]
    num_correct = [0 for _ in range(len(queries))]
    while True:
        params = SamplingParams(stop=None, temperature=args.temperature, max_tokens=2048)
        results = a.generate([prompts[i] for i in indices], params)

        next_indices = []
        for i, j in enumerate(indices):
            print(i, j)
            for output in results[i].outputs:
                prediction = extract_func(queries[j]["question"], output.text, "")
                correct = eval_answer_func({"prediction": prediction, "answer": queries[j]["answer"]})
                reward = int(correct)
                data[j].append({"response": output.text, "reward": reward})
                num_correct[j] += reward
            if num_correct[j] == 0 and N < 8:
                next_indices.append(j)
            if num_correct[j] == N and N < 4:
                next_indices.append(j)

        if len(next_indices) == 0:
            break
        M = N
        N += 1
        indices = next_indices

    total = 0
    correct = 0
    with open(args.save, "w") as f:
        for i in range(len(queries)):
            if num_correct[i] == 0 or num_correct[i] == len(data[i]):
                continue
            total += len(data[i])
            for datum in data[i]:
                correct += datum["reward"]
                f.write(json.dumps({"prompt": prompts[i], "response": datum["response"], "reward": datum["reward"]}))
                f.write("\n")
    print("{} / {}".format(correct, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["GSM8K", "MATH"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save", type=str, required=True)

    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    main()
