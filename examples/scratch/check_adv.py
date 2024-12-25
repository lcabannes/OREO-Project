import json

from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
critic = get_llm_for_sequence_regression(
    "deepseek-ai/deepseek-math-7b-instruct",
    "critic",
    normalize_reward=False,  # TODO: maybe experiment with this layer
    use_flash_attention_2=True,
    bf16=True,
    load_in_4bit=False,
    lora_rank=64,
    lora_alpha=64,
    lora_dropout=0,
    target_modules="all-linear",
    # ds_config=strategy.get_ds_train_config(is_actor=True),
    zero_init_value_head=True,
)
critic.to("cuda")

critic.load_adapter("/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_2epochs_critic", "default")

data = []
with open("/mnt/shared/phase2_test.jsonl", "r") as f:
    for line in f.readlines():
        data.append(json.loads(line))

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"

total = 0
num_correct = 0
for item in tqdm(data):
    steps = item["question"]["pre_generated_steps"]
    if item["label"]["finish_reason"] == "give_up":
        continue
    total += 1

    prompt = chat_template.format(item["question"]["problem"] + template)
    offset = len(prompt) + 1

    response = "\n".join(steps)

    text = prompt + " " + response + " " + tokenizer.eos_token
    input_token = tokenizer(text, return_tensors="pt")
    input_token.to(torch.cuda.current_device())
    with torch.no_grad():
        values = critic(**input_token, action_mask=input_token["input_ids"])

    results = []

    idx = input_token.char_to_token(offset)
    for i in range(len(steps)):
        results.append(values[0, idx - 1].item())
        offset += len(steps[i]) + 1
        idx = input_token.char_to_token(offset)
    results.append(values[0, idx - 1].item())

    correct = True
    for i, x in enumerate(item["label"]["steps"]):
        # completion = x["completions"][x["chosen_completion"]]
        completion = x["completions"][0]
        try:
            assert completion["text"] == steps[i]
        except AssertionError:
            __import__("pdb").set_trace()

        if completion["rating"] == -1:
            if results[i] <= results[i + 1] and results[i] > 0:
                correct = False
                break
    num_correct += int(correct)

print("{} / {}".format(num_correct, total))
