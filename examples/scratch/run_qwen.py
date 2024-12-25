import argparse
import json

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math


def load_queries(dataset: Dataset, input_key: str, output_key: str, extract_func):
    result = []
    for item in dataset:
        question = item[input_key]
        response = item[output_key]
        answer = extract_func(question, response, None)
        result.append({"question": question, "answer": answer})
    return result


def main():
    a = LLM(args.model, gpu_memory_utilization=0.5)
    if not args.sample:
        params = SamplingParams(args.n, stop=None, temperature=args.temperature, max_tokens=2048)
    else:
        params = SamplingParams(args.n, stop=None, top_p=0.95, top_k=16, max_tokens=2048)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    d = load_dataset("openai/gsm8k", "main")[args.split]
    queries_gsm8k = load_queries(d, "question", "answer", extract_last_single_answer)
    d = load_dataset("hendrycks/competition_math")[args.split]
    queries_math = load_queries(d, "problem", "solution", extract_math_answer)
    queries = queries_gsm8k + queries_math

    prompts = []
    for query in queries_gsm8k:
        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and produce a final answer following 4 '#', like '#### 0'",
            },
            {"role": "user", "content": query["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    for query in queries_math:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": query["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    results = a.generate(prompts, params)

    num_correct_gsm8k = 0
    num_correct_math = 0
    with open(args.save, "w") as f:
        for index, (query, result) in enumerate(zip(queries, results)):
            for output in result.outputs:
                if index < len(queries_gsm8k):
                    prediction = extract_last_single_answer(query["question"], output.text, "")
                    correct = eval_last_single_answer({"prediction": prediction, "answer": query["answer"]})
                    reward = int(correct)
                    num_correct_gsm8k += reward
                else:
                    prediction = extract_math_answer(query["question"], output.text, "")
                    correct = eval_math({"prediction": prediction, "answer": query["answer"]})
                    reward = int(correct)
                    num_correct_math += reward
                f.write(json.dumps({"prompt": prompts[index], "response": output.text, "reward": reward}))
                f.write("\n")

    print("{} / {}".format(num_correct_gsm8k, len(queries_gsm8k) * args.n))
    print(num_correct_gsm8k / len(queries_gsm8k) / args.n)
    print("{} / {}".format(num_correct_math, len(queries_math) * args.n))
    print(num_correct_math / len(queries_math) / args.n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save", type=str, required=True)

    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no_bos", action="store_true")
    args = parser.parse_args()

    main()
