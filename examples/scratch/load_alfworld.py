import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/minicpm", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)

with open("/mnt/shared/alfworld_sft.json", "r") as f:
    a = json.load(f)

data = []
for item in a:
    conversation = []
    for turn in item["conversations"]:
        conversation.append({"role": "user" if turn["from"] == "human" else "assistant", "content": turn["value"]})
    for i, turn in enumerate(conversation):
        if turn["role"] == "assistant":
            prompt = tokenizer.apply_chat_template(conversation[:i], tokenize=False, add_generation_prompt=True)
            if prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token) :].lstrip()
            response = turn["content"]
            data.append({"prompt": prompt, "response": response})

with open("/mnt/shared/annotated/minicpm_alfworld_sft.jsonl", "w") as f:
    # with open("/mnt/shared/annotated/mistral_alfworld_sft.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")
