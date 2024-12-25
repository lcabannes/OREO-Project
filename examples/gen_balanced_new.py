import argparse
import json

from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"


def load_queries(dataset: Dataset, input_key: str, output_key: str, extract_func):
    result = []
    for item in dataset:
        question = item[input_key]
        response = item[output_key]
        answer = extract_func(question, response, "")
        result.append({"question": question, "answer": answer})
    return result


def gen(queries):
    prompts = []
    for query in queries:
        q = query["question"] + template
        q: str = chat_template.format(q)
        prompts.append(q)
    return prompts


def main():
    a = LLM(args.model, tensor_parallel_size=args.tensor_parallel_size)
    params = SamplingParams(temperature=args.temperature, max_tokens=2048)

    extract_func = extract_last_single_answer if args.data == "GSM8K" else extract_math_answer
    eval_answer_func = eval_last_single_answer if args.data == "GSM8K" else eval_math

    if args.data == "GSM8K":
        d = load_dataset("openai/gsm8k", "main")[args.split]
        queries = load_queries(d, "question", "answer", extract_func)
    else:
        d = load_dataset("hendrycks/competition_math")[args.split]
        queries = load_queries(d, "problem", "solution", extract_func)
    prompts = gen(queries)

    indices = list(range(len(queries)))
    data = [[] for _ in range(len(queries))]
    num_correct = [0 for _ in range(len(queries))]
    for _ in range(16):  # TODO:
        results = a.generate([prompts[i] for i in indices], params)

        next_indices = []
        with tqdm(indices, desc="Eval") as pbar:
            for i, j in enumerate(pbar):
                output = results[i].outputs[0]  # only 1 output is generated for each question
                prediction = extract_func(queries[j]["question"], output.text, "")
                correct = eval_answer_func({"prediction": prediction, "answer": queries[j]["answer"]})
                reward = int(correct)
                data[j].append({"response": output.text, "reward": reward})
                num_correct[j] += reward

                if num_correct[j] < args.response_count or len(data[j]) - num_correct[j] < args.response_count:
                    next_indices.append(j)
                # TODO: stoping criteria?

                indices = next_indices

    total = 0
    correct = 0
    with open(args.save, "w") as f:
        for i in range(len(queries)):
            correct_indices = []
            incorrect_indices = []
            for j in range(len(data[i])):
                if data[i][j]["reward"] == 1:
                    correct_indices.append(j)
                else:
                    incorrect_indices.append(j)
            N = len(correct_indices)
            M = len(incorrect_indices)
            N = min(N, max(M, 1))
            indices = np.random.choice(correct_indices, min(N, args.response_count), replace=False).tolist()
            indices.extend(np.random.choice(incorrect_indices, min(M, args.response_count), replace=False).tolist())
            total += len(indices)
            for j in indices:
                correct += data[i][j]["reward"]
                f.write(
                    json.dumps(
                        {"prompt": prompts[i], "response": data[i][j]["response"], "reward": data[i][j]["reward"]}
                    )
                )
                f.write("\n")

    print("{} / {}".format(correct, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["GSM8K", "MATH"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--response_count", type=int, default=4)
    args = parser.parse_args()

    main()
