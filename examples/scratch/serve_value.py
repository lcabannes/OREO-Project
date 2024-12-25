from typing import List
import argparse
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression

add_eos = False


class GetValuesReq(BaseModel):
    states: List[str | List[dict]]


class Model:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.critic = get_llm_for_sequence_regression(
            model,
            "critic",
            normalize_reward=False,  # TODO: maybe experiment with this layer
            use_flash_attention_2=True,
            bf16=True,
            load_in_4bit=False,
            # lora_rank=64,
            # lora_alpha=64,
            # lora_dropout=0,
            # target_modules="all-linear",
            # ds_config=strategy.get_ds_train_config(is_actor=True),
        )
        self.critic.to(torch.cuda.current_device())
        self.critic.eval()
        self.mutex = Lock()


app = FastAPI()
model = Model(
    "/mnt/data/ckpt/pcl/minicpm_full_alfworld_pcl_token-level_beta0-03_actor-loss-dro_kl-reg-unbiased1e-2_plot-weights_3epochsf_critic"
)  # TODO:


@app.post("/get_values")
def get_values(req: GetValuesReq):
    N = len(req.states)
    bsz = 8
    results = []
    for i in range(0, N, bsz):
        # batch_str = [s + " " + model.tokenizer.eos_token for s in req.states[i : i + bsz]]  # TODO: add eos?
        batch_str = []
        for state in req.states[i : i + bsz]:
            if isinstance(state, str):
                if add_eos:
                    batch_str.append(state + " " + model.tokenizer.eos_token)
                else:
                    batch_str.append(state)
            elif isinstance(state, list):
                s = model.tokenizer.apply_chat_template(state, tokenize=False)
                if add_eos:
                    s = s + " " + model.tokenizer.eos_token
                batch_str.append(s)
            else:
                raise f"unknown type: {type(state)}"
        model.mutex.acquire()
        try:
            input_token = model.tokenizer(batch_str, padding=True, return_tensors="pt")
            input_token.to(torch.cuda.current_device())
            with torch.no_grad():
                values: torch.Tensor = model.critic(**input_token, action_mask=input_token["input_ids"])
                results.extend(values[:, -1].tolist())
        except Exception as e:
            raise e
        finally:
            model.mutex.release()
    return results


@app.get("/test")
def test():
    return [1, 2, 3]
