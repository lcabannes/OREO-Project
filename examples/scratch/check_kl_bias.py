from tqdm import tqdm
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from openrlhf.datasets.pcl_dataset import PCLDataset

base_model_path = "deepseek-ai/deepseek-math-7b-instruct"
pcl_model_path = "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(pcl_model_path, trust_remote_code=True)
    .to(torch.cuda.current_device())
    .to(torch.bfloat16)
)
ref_model = (
    AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    .to(torch.cuda.current_device())
    .to(torch.bfloat16)
)
dataset = PCLDataset(tokenizer, 2048, train_file="/mnt/shared/annotated/train-dsm-b2.jsonl", step_level=True)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)


def get_logps(model, ids, masks):
    outputs = model(ids, attention_mask=masks)
    logits = outputs["logits"][:, :-1, :]
    labels = ids[:, 1:]
    logps = torch.log_softmax(logits, dim=-1)
    logps_raw = logps
    logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logps, logps_raw


biased_kls = []
unbiased_kls = []

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

            # estimate KL
            kls = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
            biased_kl_estimate = (kls * action_masks[:, 1:]).sum(dim=-1) / action_masks[:, 1:].sum(dim=-1)
            biased_kls.extend(biased_kl_estimate.tolist())

            tmp = F.kl_div(reference_logps_raw, logps_raw, log_target=True, reduction="none").sum(dim=-1)
            kl_estimate = (tmp * action_masks[:, 1:]).sum(dim=-1) / action_masks[:, 1:].sum(dim=-1)
            unbiased_kls.extend(kl_estimate.tolist())

            cnt += 1
            if cnt == 1000:
                break

torch.save(biased_kls, "biased_kls.pt")
torch.save(unbiased_kls, "unbiased_kls.pt")
