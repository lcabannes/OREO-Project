import json

from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = []

d = load_dataset("openai/gsm8k", "main")

for item in d["train"]:
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and produce a final answer following 4 '#', like '#### 0'",
        },
        {"role": "user", "content": item["question"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    data.append({"prompt": prompt, "response": item["answer"]})

d = load_dataset("hendrycks/competition_math")

for item in d["train"]:
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": item["problem"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    data.append({"prompt": prompt, "response": item["solution"]})

with open("/mnt/shared/annotated/train-qwen-sft.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")
