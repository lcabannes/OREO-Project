import json

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_math

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"
np.random.seed(42)

# d_gsm8k = load_dataset("openai/gsm8k", "main")
# solution_gsm8k = {}
# for item in d_gsm8k["train"]:
#     solution_gsm8k[item["question"].strip()] = extract_last_single_answer(item["question"], item["answer"], "")

# d_math = load_dataset("hendrycks/competition_math")
# solution_math = {}
# for item in d_math["train"]:
#     solution_math[item["problem"].strip()] = extract_math_answer(item["problem"], item["solution"], "")

# print(1)

# d = load_dataset("hkust-nlp/dart-math-hard")
# for item in tqdm(d["train"]):
#     q = item["query"].strip()
#     if not q in solution_gsm8k and not q in solution_math:
#         __import__("pdb").set_trace()

d = load_dataset("hkust-nlp/dart-math-hard")
N = len(d["train"])
indices = np.random.choice(N, N // 5, replace=False).tolist()

with open("/mnt/shared/annotated/train-dsm-dart-math.jsonl", "w") as f:
    # for item in tqdm(d["train"]):
    for i in indices:
        item = d["train"][i]
        f.write(
            json.dumps(
                {
                    "prompt": chat_template.format(item["query"] + template),
                    "response": " " + item["response"],
                    "reward": 1,
                }
            )
        )
        f.write("\n")
