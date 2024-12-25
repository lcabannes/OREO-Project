import argparse
import json

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

chat_template = "User: {}\n\nAssistant:"


def load_queries(dataset: Dataset, input_key: str, output_key: str):
    result = []
    for item in dataset:
        question = item[input_key]
        response = item[output_key]
        answer = extract_math_answer(question, response, None)
        result.append({"question": question, "answer": answer})
    return result


def main():
    a = LLM(args.model)
    params = SamplingParams(args.n, stop=None, temperature=args.temperature, max_tokens=2048)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    template = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    d = load_dataset("openai/gsm8k", "main")[args.split]
    queries_gsm8k = load_queries(d, "question", "answer")
    d = load_dataset("hendrycks/competition_math")[args.split]
    queries_math = load_queries(d, "problem", "solution")
    queries = queries_gsm8k + queries_math

    # accs = []
    # for query in queries[:100]:
    #     q = query["question"] + template
    #     q: str = tokenizer.apply_chat_template(
    #         [{"role": "user", "content": q}], add_generation_prompt=True, add_special_tokens=False, tokenize=False
    #     )
    #     result = a.generate(q, sampling_params=params)[0]
    #     acc = 0.0
    #     for output in result.outputs:
    #         prediction = extract_math_answer(query["question"], output.text, "")
    #         correct = eval_math({"prediction": prediction, "answer": query["answer"]})
    #         acc += correct / len(result.outputs)
    #     accs.append(acc)

    prompts = []
    for query in queries:
        q = query["question"] + template
        if args.no_bos:
            q = chat_template.format(q)
        else:
            q: str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True, add_special_tokens=False, tokenize=False
            )
        prompts.append(q)
    results = a.generate(prompts, sampling_params=params)

    num_correct_gsm8k = 0
    num_correct_math = 0
    with open(args.save, "w") as f:
        for index, (query, result) in enumerate(zip(queries, results)):
            for output in result.outputs:
                prediction = extract_math_answer(query["question"], output.text, "")
                correct = eval_math({"prediction": prediction, "answer": query["answer"]})
                reward = int(correct)
                if index < len(queries_gsm8k):
                    num_correct_gsm8k += reward
                else:
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
    parser.add_argument("--no_bos", action="store_true")
    args = parser.parse_args()

    main()
