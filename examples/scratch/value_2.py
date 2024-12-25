from colorama import Fore, Back, Style
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.pcl_dataset import PCLDataset

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
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_critic-only_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_step-level_kl-reg-unbiased1e-2_actor-freeze-ratio0-2_critic",
#     "default",
# )
critic.load_adapter(
    "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg-unbiased1e-2_actor-freeze-ratio0-2_critic",
    "default",
)
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_step-level_critic-only_critic",
#     "default",
# )

dataset = PCLDataset(
    tokenizer,
    2048,
    train_file="/mnt/shared/annotated/train-dsm-b2.jsonl",
    step_level=True,
    padding_side="left",
)
sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42, drop_last=True)
loader = DataLoader(dataset, 8, sampler=sampler, drop_last=True, collate_fn=dataset.collate_fn)

results = []
for cnt, (ids, masks, state_masks, action_masks, rewards) in enumerate(tqdm(loader)):
    if cnt == 100:
        break
    ids = ids.squeeze(1).to(torch.cuda.current_device())
    masks = masks.squeeze(1).to(torch.cuda.current_device())
    state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
    action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
    rewards = rewards.squeeze(1).to(torch.cuda.current_device())

    with torch.no_grad():
        values: torch.Tensor = critic(ids, attention_mask=masks, action_mask=action_masks)

    for i in range(values.shape[0]):
        if rewards[i].item() == -1:
            continue
        last_idx = (ids[i] == tokenizer.bos_token_id).nonzero().item()
        tmp = float("-inf")
        for j in state_masks[i].nonzero().flatten().tolist():
            # print(tokenizer.decode(ids[i, last_idx:j]))
            # print(Fore.WHITE + Back.GREEN + "--- Value: {}".format(values[i, j].item()), end="")
            # print(Style.RESET_ALL)
            last_idx = j + 1
            tmp = max(tmp, values[i, j].item())
        results.append(tmp)
        # print("Reward: {}".format(rewards[i].item()))
        # __import__("pdb").set_trace()
torch.save(results, "max_step_value_token.pt")
