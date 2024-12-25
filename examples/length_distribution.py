import json

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/minicpm", trust_remote_code=True)

Ls = []
with open("/mnt/shared/annotated/train-n5-t1.jsonl", "r") as f:
    for line in f.readlines():
        tmp = json.loads(line)
        text = tmp["prompt"] + tmp["response"]
        inputs = tokenizer(text, return_tensors="pt")
        Ls.append(inputs["input_ids"].shape[1])
plt.hist(Ls, bins=40)
plt.savefig("./tmp.jpg")
