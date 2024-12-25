import json
import itertools
import copy

import numpy as np
from datasets import load_dataset

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"
np.random.seed(42)

cnt = 60000

d = load_dataset("hkust-nlp/dart-math-hard")
N = len(d["train"])

explored_data_pos = []
explored_data_neg = []
with open("/mnt/shared/annotated/train-dsm-b2.jsonl", "r") as f:
    for line in f.readlines():
        tmp = json.loads(line)
        if tmp["reward"] == 1:
            explored_data_pos.append(tmp)
        else:
            explored_data_neg.append(tmp)

assert len(explored_data_pos) < cnt
assert len(explored_data_neg) < cnt

data = copy.deepcopy(explored_data_pos) + copy.deepcopy(explored_data_neg)
indices = np.random.choice(N, cnt - len(explored_data_pos), replace=False).tolist()
for i in indices:
    item = d["train"][i]
    data.append(
        {
            "prompt": chat_template.format(item["query"] + template),
            "response": " " + item["response"],
            "reward": 1,
        }
    )

indices = np.random.choice(len(explored_data_neg), cnt - len(explored_data_neg), replace=True).tolist()

for i in indices:
    data.append(explored_data_neg[i])

with open("/mnt/shared/annotated/train-dsm-dart-math-b.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")
