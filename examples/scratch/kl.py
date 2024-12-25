from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from openrlhf.datasets.pcl_dataset import PCLDataset

base_model_path = "./ckpt/minicpm_sft_gsm8k_merged"
pcl_model_path = "./ckpt/a_lr1e-4_actor-lr5e-5_merged"
# base_model_path = "deepseek-ai/deepseek-math-7b-instruct"
# pcl_model_path = "/mnt/data/ckpt/dsm-inst_lr1e-4_actor-lr5e-5_merged"
# base_model_path = "/mnt/data/ckpt/mistral_sft_gsm8k_10epochs_merged"
# pcl_model_path = "/mnt/data/ckpt/mistral_t0.3_beta0.3_lr1e-4_actor-lr5e-5"
# base_model_path = "deepseek-ai/deepseek-math-7b-instruct"
# pcl_model_path = "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_merged"

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
dataset = PCLDataset(tokenizer, 2048, step_level=True)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)


def get_logps(model, ids, masks):
    outputs = model(ids, attention_mask=masks)
    logits = outputs["logits"][:, :-1, :]
    labels = ids[:, 1:]
    logps = torch.log_softmax(logits, dim=-1)
    logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logps


results = []
with torch.no_grad():
    cnt = 0
    with tqdm(loader) as pbar:
        for ids, masks, state_masks, action_masks, rewards in pbar:
            ids = ids.squeeze(1).to(torch.cuda.current_device())
            masks = masks.squeeze(1).to(torch.cuda.current_device())
            state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
            action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = rewards.squeeze(1).to(torch.cuda.current_device())

            logps = get_logps(model, ids, masks)
            reference_logps = get_logps(ref_model, ids, masks)

            # estimate KL
            kls = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
            kl_estimate = torch.mean((kls * action_masks[:, 1:]).sum(dim=-1) / action_masks[:, 1:].sum(dim=-1))

            results.append(kl_estimate.item())
            pbar.set_description("KL estimate: {:.3f}".format(np.mean(results)))
            cnt += 1
            if cnt == 1000:
                break

print(np.mean(results))
print(np.std(results))
plt.hist(results, bins=1000)
plt.xlim(0, 5)
plt.savefig(f"kl-3.jpg")
