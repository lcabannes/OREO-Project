from typing import Optional
import torch
from torch import nn
from transformers import AutoConfig

from .actor import Actor, log_probs_from_logits


class ActorCritic(Actor):
    def __init__(self, pretrain_or_model, *args, **kwargs):
        super().__init__(pretrain_or_model, *args, **kwargs)

        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        bf16 = kwargs.get("bf16")
        self.value_head = nn.Linear(
            config.hidden_size, 1, device=self.model.device, dtype=torch.bfloat16 if bf16 else None
        )

    def forward(
        self,
        sequences: torch.LongTensor = None,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        output_hidden_states=False,
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]
