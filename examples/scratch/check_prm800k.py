import json

from tqdm import tqdm
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

cnt = 0
for item in tqdm(data):
    steps = item["question"]["pre_generated_steps"]
    if item["label"]["finish_reason"] == "give_up":
        continue
    cnt += 1
    for i, x in enumerate(item["label"]["steps"]):
        # completion = x["completions"][x["chosen_completion"]]
        completion = x["completions"][0]
        try:
            assert completion["text"] == steps[i]
        except AssertionError:
            __import__("pdb").set_trace()

print(cnt)

tmp = set(item["label"]["finish_reason"] for item in data)
print(tmp)
