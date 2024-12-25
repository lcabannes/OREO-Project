import re
import json


def extract_answer(answer: str):
    result = re.search("#### (.+)", answer.strip())
    if result is None:
        # __import__("pdb").set_trace()
        return ""
    else:
        return result.group(1)


with open("/mnt/shared/annotated/train-mix.jsonl", "w") as f_out:
    with open("/mnt/shared/annotated/train-mistral-n5-t0.3.jsonl", "r") as f:
        for line in f.readlines():
            f_out.write(line)

    cnt = 0
    with open("/mnt/shared/annotated/train-n5-t1.jsonl", "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            resp = item["response"]
            answer = extract_answer(resp)
            if answer == "":
                cnt += 1
                f_out.write(line)

print(cnt)
