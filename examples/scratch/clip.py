import torch
from torch.functional import F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from openrlhf.datasets.pcl_dataset import PCLDataset
from openrlhf.utils import get_tokenizer

model = AutoModelForCausalLM.from_pretrained("/mnt/data/ckpt/dsm-inst_b2_rej-sampling_10epochs")
model.to(torch.cuda.current_device())
ref_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
ref_model.to(torch.cuda.current_device())
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")

train_dataset = PCLDataset(
    tokenizer,
    2048,
    train_file="/mnt/shared/annotated/test-dsm-inst_b2_rej-sampling_10epochs_merged.jsonl",
    step_level=False,
    padding_side="left",
)

loader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)

results = []
cnt = 0
with tqdm(loader) as pbar:
    for ids, masks, state_masks, action_masks, rewards in pbar:
        cnt += 1
        if cnt > 1000:
            break
        ids = ids.squeeze(1).to(torch.cuda.current_device())
        masks = masks.squeeze(1).to(torch.cuda.current_device())
        state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
        action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
        rewards = rewards.squeeze(1).to(torch.cuda.current_device())

        with torch.no_grad():
            position_ids = masks.long().cumsum(-1) - 1
            position_ids.masked_fill_(masks == 0, 1)

            outputs = model(ids, attention_mask=masks, position_ids=position_ids)
            logits = outputs["logits"]
            logits = logits[:, :-1, :]

            outputs_ref = ref_model(ids, attention_mask=masks, position_ids=position_ids)
            target = outputs_ref["logits"]
            target = target[:, :-1, :]

        kls = F.kl_div(target.log_softmax(dim=-1), logits.log_softmax(dim=-1), reduction="none", log_target=True)
        kls = kls.max(dim=-1).values.flatten().tolist()
        flags = state_masks[:, :-1].flatten().tolist()
        kls_filtered = filter(lambda x: bool(x[0]), zip(state_masks[:, :-1].flatten().tolist(), kls))
        results.extend(list(map(lambda x: x[1], kls_filtered)))

        pbar.set_description("mean kl: {:.2f}".format(np.mean(results)))

torch.save(results, "kls-sft.pt")
plt.hist(results, bins=100, density=True)
plt.savefig("kls-sft.jpg")
