import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/ckpt/pcl/minicpm_alfworld_sft_full")

with open("/mnt/shared/annotated/annotated/train-n5-t1-minicpm-alfworld.json", "r") as f:
    data = json.load(f)

sft_data = []
for item in data:
    if int(item["reward"]) != 1:
        continue
    conversation = item["conversations"]
    for i, turn in enumerate(conversation):
        if turn["role"] == "assistant":
            prompt = tokenizer.apply_chat_template(conversation[:i], tokenize=False, add_generation_prompt=True)
            if prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token) :].lstrip()
            response = turn["content"]
            sft_data.append({"prompt": prompt, "response": response})

with open("/mnt/shared/annotated/train-alfworld-filtered.jsonl", "w") as f:
    for item in sft_data:
        f.write(json.dumps(item))
        f.write("\n")
