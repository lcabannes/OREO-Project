import json

from tqdm import tqdm
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = []

with open("/mnt/shared/annotated/train-dsm-n5-t1.jsonl", "r") as f:
    for line in f.readlines():
        data.append(json.loads(line))

with open("/mnt/shared/annotated/train-qwen-sft-from-dsm.jsonl", "w") as f:
    for item in tqdm(data):
        if item["reward"] == 0:
            continue
        problem = item["prompt"][len("<\uff5cbegin\u2581of\u2581sentence\uff5c>User: ") :][
            : -len("\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:")
        ]
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = item["response"].lstrip(" ")
        f.write(json.dumps({"prompt": prompt, "response": response}))
        f.write("\n")
