import json

from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
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
# with open("/mnt/shared/annotated/test-dsm-inst_b2_lr1e-4_actor-lr5e-5_2epochs_merged.jsonl", "r") as f:
with open("/mnt/shared/annotated/train-dsm-b2.jsonl", "r") as f:
    for line in f.readlines():
        data.append(json.loads(line))

bsz = 8

result = 0
# data = data[1319:]
for i in tqdm(range(0, len(data), bsz)):
    batch = [x["prompt"] + x["response"] + " " + tokenizer.eos_token for x in data[i : i + bsz]]
    input_token = tokenizer(batch, max_length=2048, padding=True, truncation=True, return_tensors="pt")
    input_token.to("cuda")
    with torch.no_grad():
        values = critic(**input_token, action_mask=input_token["input_ids"])
        for j in range(len(batch)):
            if data[i + j]["reward"] == 1:
                correct = int(values[j, -2].item() > 0)
            else:
                correct = int(values[j, -2].item() < 0)
            # if correct == 0:
            #     __import__("pdb").set_trace()
            result += correct

print("{} / {}".format(result, len(data)))
