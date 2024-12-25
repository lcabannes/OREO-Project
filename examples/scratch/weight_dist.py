from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from openrlhf.datasets.pcl_dataset import PCLDataset

base_model_name = "/mnt/data/ckpt/qwen_sft"
model_name = "/mnt/data/ckpt/qwen_full_lr5e-6_actor-lr5e-7_beta0-03_rew01_actor-loss-dro"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.cuda.current_device()).to(torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(torch.cuda.current_device()).to(torch.bfloat16)

dataset = PCLDataset(
    tokenizer, 2048, train_file="/mnt/shared/annotated/train-qwen.jsonl", padding_side="left", rew_mul=1, rew_add=0
)
sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42, drop_last=True)
loader = DataLoader(dataset, 2, sampler=sampler, drop_last=True, collate_fn=dataset.collate_fn)


def get_logps(model, ids, masks):
    outputs = model(ids, attention_mask=masks)
    logits = outputs["logits"][:, :-1, :]
    labels = ids[:, 1:]
    logps = torch.log_softmax(logits, dim=-1)
    logps_raw = logps
    logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logps, logps_raw


weights_correct = []
weights_incorrect = []
with torch.no_grad():
    cnt = 0
    with tqdm(loader) as pbar:
        for ids, masks, state_masks, action_masks, rewards in pbar:
            ids = ids.squeeze(1).to(torch.cuda.current_device())
            masks = masks.squeeze(1).to(torch.cuda.current_device())
            state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
            action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = rewards.squeeze(1).to(torch.cuda.current_device())

            logps, logps_raw = get_logps(model, ids, masks)
            reference_logps, reference_logps_raw = get_logps(ref_model, ids, masks)

            tmp = torch.masked_fill(logps - reference_logps, ~action_masks[:, 1:].bool(), 0).sum(dim=-1)
            weights = tmp.exp()
            for i in range(rewards.shape[0]):
                if rewards[i].item() == 1:
                    weights_correct.append(weights[i].item())
                else:
                    weights_incorrect.append(weights[i].item())

            cnt += 1
            if cnt == 1000:
                break

torch.save([weights_correct, weights_incorrect], "weights_actor-lr5e-7.pt")
# plt.hist(weights_correct, bins=100, label="correct")
# plt.hist(weights_incorrect, bins=100, label="correct")
# plt.savefig("weights.jpg")
