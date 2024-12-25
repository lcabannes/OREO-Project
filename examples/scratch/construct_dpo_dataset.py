import json

import numpy as np

# name = "qwen"
# sample_cnt = 6
name = "dsm-b2"
sample_cnt = 10

data = []
with open(f"/mnt/shared/annotated/train-{name}.jsonl", "r") as f:
    for line in f.readlines():
        data.append(json.loads(line))

s = set()


def sample_pairs(pos_resp, neg_resp):
    if len(pos_resp) == 0 or len(neg_resp) == 0:
        return []
    indices = []
    for i in range(len(pos_resp)):
        for j in range(len(neg_resp)):
            indices.append((i, j))
    tmp = np.random.choice(range(len(indices)), min(sample_cnt, len(indices)), replace=False)
    indices = [indices[i] for i in tmp]
    results = []
    for i, j in indices:
        results.append(
            {"prompt": pos_resp[i]["prompt"], "chosen": pos_resp[i]["response"], "rejected": neg_resp[j]["response"]}
        )
    return results


previous_prompt = None
paired_data = []
for item in data:
    if item["prompt"] != previous_prompt:
        if previous_prompt is not None:
            s.add(previous_prompt)
            paired_data.extend(sample_pairs(pos_resp, neg_resp))
        assert item["prompt"] not in s
        previous_prompt = item["prompt"]
        pos_resp = []
        neg_resp = []
    if int(item["reward"]) == 1:
        pos_resp.append(item)
    else:
        neg_resp.append(item)

paired_data.extend(sample_pairs(pos_resp, neg_resp))

with open(f"/mnt/shared/annotated/train-{name}-dpo.jsonl", "w") as f:
    for item in paired_data:
        f.write(json.dumps(item))
        f.write("\n")
