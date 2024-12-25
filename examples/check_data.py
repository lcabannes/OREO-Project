import json

import torch
from transformers import AutoTokenizer

data = []

with open("/mnt/shared/annotated/train-n5-t1.jsonl", "r") as f:
    for line in f.readlines():
        data.append(json.loads(line))

step_level = True
max_length = 2048
tokenizer = AutoTokenizer.from_pretrained("/mnt/minicpm", trust_remote_code=True)

for index in range(len(data)):
    # prompt = data[index]["prompt"]
    # response: str = data[index]["response"]
    prompt = "Hello, world!\n"
    response = "asdf\n\naaaa\nf"
    __import__("pdb").set_trace()
    reward = data[index]["reward"]
    if step_level:
        lines = response.split("\n")
        response = "\n".join(lines)

    input_token = tokenizer(
        prompt + response + " " + tokenizer.eos_token,
        max_length=max_length,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )  # TODO
    ids = input_token["input_ids"]
    attention_mask = input_token["attention_mask"]

    state_mask = torch.zeros_like(ids)
    action_mask = torch.zeros_like(ids)
    if not step_level:
        # TODO: this are masks for token-wise PCL
        idx = input_token.char_to_token(len(prompt))  # first token pos of response
        state_mask[0][idx - 1 : -1] = 1
        action_mask[0][idx:] = 1
    else:
        offset = len(prompt)
        for i, line in enumerate(lines):
            idx = input_token.char_to_token(offset)
            if idx is None:
                __import__("pdb").set_trace()
            state_mask[0][idx - 1] = 1
            offset += len(line) + 1
            if i == len(lines) - 1:
                action_mask[0][idx:] = 1
            else:
                next_idx = input_token.char_to_token(offset)
                action_mask[0][idx:next_idx] = 1
