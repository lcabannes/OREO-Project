import re
import json

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams


def extract_answer(answer: str):
    result = re.search("#### (.+)", answer.strip())
    if result is None:
        # __import__("pdb").set_trace()
        return ""
    else:
        return result.group(1)


def main():
    a = LLM(args.model, trust_remote_code=True)
    params = SamplingParams(args.n, stop=None, temperature=args.temperature, max_tokens=2048)

    d = load_dataset("openai/gsm8k", "main")
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response: Let's think step by step.\n"

    d = d[args.split]
    prompts = []
    for item in d:
        question = item["question"]
        prompts.append(template.format(question))

    resp = a.generate(prompts, sampling_params=params)

    num_correct = 0
    num_wrong_format = 0
    with open(args.save, "w") as f:
        for i in range(len(resp)):
            for j in range(len(resp[i].outputs)):
                result = resp[i].outputs[j].text
                ai_answer = extract_answer(result)
                gt_answer = extract_answer(d[i]["answer"])
                assert gt_answer != ""
                reward = int(ai_answer == gt_answer)
                num_correct += reward
                if ai_answer == "":
                    num_wrong_format += 1
                f.write(json.dumps({"prompt": prompts[i], "response": result, "reward": reward}))
                f.write("\n")
    print("{} / {}".format(num_correct, len(resp) * args.n))
    print(num_wrong_format)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save", type=str, required=True)

    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)

    args = parser.parse_args()

    main()
