import json
import random

name = "dsm-b2"
with open(f"test-qwen.jsonl", "r") as f:
    with open(f"filtered_test-qwen.jsonl", "w") as f_out:
        results = []
        for line in f.readlines():
            tmp = json.loads(line)
            if tmp["reward"] == 1:
                results.append(line)
        random.shuffle(results)
        for line in results:
            f_out.write(line)
