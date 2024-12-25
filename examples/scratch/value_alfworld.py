import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.pcl_dataset_alfworld import PCLDatasetAlfWorld

tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/mistral_alfworld_pcl")
critic = get_llm_for_sequence_regression(
    "/mnt/lustre/ckpt/eto-mistral-alfworld-sft",
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
# critic.load_adapter("/mnt/lustre/ckpt/mistral_alfworld_pcl_critic", "default")
# critic.load_adapter("/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg10_critic", "default")
critic.load_adapter("/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg100_2epochs_critic", "default")
# critic.load_adapter("/mnt/lustre/ckpt/mistral_alfworld_pcl_critic-only_critic", "default")

dataset = PCLDatasetAlfWorld(
    tokenizer, 4096, train_file="/mnt/shared/annotated/train-alfworld.json", step_level=True, padding_side="right"
)
loader = DataLoader(dataset, 1, collate_fn=dataset.collate_fn)

# for ids, masks, state_masks, action_masks, rewards in loader:
#     ids = ids.squeeze(1).to(torch.cuda.current_device())
#     masks = masks.squeeze(1).to(torch.cuda.current_device())
#     state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
#     action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
#     rewards = rewards.squeeze(1).to(torch.cuda.current_device())

#     values = critic(ids, attention_mask=masks, action_mask=action_masks)
#     results = []
#     for i in range(values.shape[1]):
#         if state_masks[0, i].item():
#             results.append(values[0, i].item())

#     print(tokenizer.decode(ids[0]))
#     __import__("pdb").set_trace()

for i in range(len(dataset)):
    ids, masks, state_masks, action_masks, rewards = dataset.collate_fn([dataset[i]])
    ids = ids.squeeze(1).to(torch.cuda.current_device())
    masks = masks.squeeze(1).to(torch.cuda.current_device())
    state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
    action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
    rewards = rewards.squeeze(1).to(torch.cuda.current_device())

    values = critic(ids, attention_mask=masks, action_mask=action_masks)
    results = []
    for j in range(values.shape[1]):
        if state_masks[0, j].item():
            results.append(values[0, j].item())

    print(tokenizer.decode(ids[0]))
    k = 0
    for j in range(3, len(dataset.data[i]["conversations"])):
        tmp = dataset.data[i]["conversations"][j]
        print(tmp["content"])
        if tmp["role"] == "assistant":
            print(results[k])
            k += 1
            __import__("pdb").set_trace()
