from abc import ABC

import numpy as np
import torch
import torch.distributed
from torch.functional import F
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm


class PCLTrainer(ABC):
    def __init__(
        self,
        model,
        critic,
        ref_model,
        ema_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        critic_optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        critic_scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        critic_only: bool = False,
        step_level: bool = False,
        single_step_actor_loss: bool = False,
        dro_actor_loss: bool = False,
        traj_actor_loss: bool = False,
        clip_log: float | None = None,
        ema_beta: float = 0.995,
        kl_reg: float | None = None,
        kl_targ: float | None = None,
        init_log_kl_coeff: float = 0.0,
        kl_coeff_lr: float = 1e-3,
        unbiased_kl: bool = False,
        forward_kl: bool = False,
        hinge_coeff: float | None = None,
        actor_freeze_steps: int | None = None,
        importance_sampling: bool = False,
        importance_sampling_2: bool = False,
        plot_weights: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.critic = critic
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.ema_model = ema_model
        self.scheduler = scheduler
        self.critic_scheduler = critic_scheduler
        self.optimizer = optim
        self.critic_optimizer = critic_optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        self.beta = beta

        # TODO: Mixtral 8*7b
        # self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.critic_only = critic_only
        self.step_level = step_level
        self.single_step_actor_loss = single_step_actor_loss
        self.dro_actor_loss = dro_actor_loss
        self.traj_actor_loss = traj_actor_loss
        self.clip_log = clip_log
        self.ema_beta = ema_beta
        self.kl_reg = kl_reg
        self.kl_targ = kl_targ
        self.unbiased_kl = unbiased_kl
        self.forward_kl = forward_kl
        self.hinge_coeff = hinge_coeff
        self.actor_freeze_steps = actor_freeze_steps
        self.importance_sampling = importance_sampling
        self.importance_sampling_2 = importance_sampling_2
        self.plot_weights = plot_weights
        if self.kl_targ:
            self.log_kl_coeff = torch.tensor(init_log_kl_coeff, device=torch.cuda.current_device(), requires_grad=True)
            self.kl_coeff_optim = torch.optim.Adam([self.log_kl_coeff], lr=kl_coeff_lr)

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            self.critic.train()
            self.ref_model.eval()
            if self.ema_model is not None:
                self.ema_model.eval()
            loss_mean = 0
            actor_loss_mean = 0
            pos_log_weights = []
            neg_log_weights = []
            # train
            for ids, masks, state_masks, action_masks, rewards in self.train_dataloader:
                ids = ids.squeeze(1).to(torch.cuda.current_device())
                masks = masks.squeeze(1).to(torch.cuda.current_device())
                state_masks = state_masks.squeeze(1).to(torch.cuda.current_device())
                action_masks = action_masks.squeeze(1).to(torch.cuda.current_device())
                rewards = rewards.squeeze(1).to(torch.cuda.current_device())

                with torch.no_grad():
                    model = self.ema_model if self.ema_model is not None else self.model
                    accumulated_logps, logps, _ = self.accumulated_logps(model, ids, masks, action_masks)
                values: torch.Tensor = self.critic(ids, attention_mask=masks, action_mask=action_masks)
                values_detached = values.clone().detach()

                with torch.no_grad():
                    reference_accumulated_logps, reference_logps, reference_logps_raw = self.accumulated_logps(
                        self.ref_model, ids, masks, action_masks
                    )

                if self.importance_sampling:
                    with torch.no_grad():
                        weights, log_weights = self.weights(logps, reference_logps, action_masks)
                else:
                    weights = None

                # estimate KL
                kls = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
                kl_estimate = torch.mean((kls * action_masks[:, 1:]).sum(dim=-1) / action_masks[:, 1:].sum(dim=-1))

                # value statistics
                max_value = values_detached.masked_fill(~state_masks[:, :-1].bool(), float("-inf")).max()
                min_value = values_detached.masked_fill(~state_masks[:, :-1].bool(), float("inf")).min()

                loss, clip_ratio = self.loss(
                    accumulated_logps,
                    reference_accumulated_logps,
                    values,
                    state_masks,
                    rewards,
                    self.clip_log,
                    weights,
                )
                self.strategy.backward(loss, self.critic, self.critic_optimizer)
                self.strategy.optimizer_step(self.critic_optimizer, self.critic, self.critic_scheduler)

                train_actor = not self.critic_only and not (
                    self.actor_freeze_steps is not None
                    and global_step <= self.actor_freeze_steps * self.strategy.accumulated_gradient
                )
                if train_actor:
                    accumulated_logps, logps, logps_raw = self.accumulated_logps(self.model, ids, masks, action_masks)
                    # with torch.no_grad():
                    #     values = self.critic(ids, attention_mask=masks, action_mask=action_masks)

                    # with torch.no_grad():
                    #     reference_accumulated_logps = self.accumulated_logps(self.ref_model, ids, masks, action_masks)

                    if self.single_step_actor_loss:
                        actor_loss = self.single_step_loss(
                            logps,
                            reference_logps,
                            accumulated_logps,
                            reference_accumulated_logps,
                            values_detached,
                            state_masks,
                            rewards,
                        )
                    elif self.dro_actor_loss:
                        actor_loss = self.dro_loss(
                            logps,
                            reference_logps,
                            accumulated_logps,
                            reference_accumulated_logps,
                            values_detached,
                            state_masks,
                            rewards,
                            weights,
                        )
                    elif self.traj_actor_loss:
                        actor_loss, vs0 = self.traj_loss(
                            accumulated_logps, reference_accumulated_logps, values_detached, state_masks, rewards
                        )
                    else:
                        actor_loss = self.loss(
                            accumulated_logps, reference_accumulated_logps, values_detached, state_masks, rewards
                        )

                    if self.kl_reg is not None or self.kl_targ is not None:
                        if not self.unbiased_kl:
                            kls_actor = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
                        elif not self.forward_kl:
                            kls_actor = F.kl_div(
                                reference_logps_raw, logps_raw, reduction="none", log_target=True
                            ).sum(dim=-1)
                        else:
                            kls_actor = F.kl_div(
                                logps_raw, reference_logps_raw, reduction="none", log_target=True
                            ).sum(dim=-1)
                        kl_estimate_actor = torch.mean(
                            (kls_actor * action_masks[:, 1:]).sum(dim=-1) / action_masks[:, 1:].sum(dim=-1)
                        )
                        kl_reg = self.kl_reg if self.kl_reg is not None else self.log_kl_coeff.exp().detach()
                        self.strategy.backward(actor_loss + kl_reg * kl_estimate_actor, self.model, self.optimizer)
                    else:
                        self.strategy.backward(actor_loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                    if self.kl_targ is not None:
                        kl_delta = (kl_estimate_actor - self.kl_targ).detach()
                        torch.distributed.all_reduce(kl_delta, torch.distributed.ReduceOp.AVG)
                        kl_coeff_loss = (-self.log_kl_coeff * kl_delta) / self.strategy.accumulated_gradient
                        kl_coeff_loss.backward()
                        if global_step % self.strategy.accumulated_gradient == 0:
                            self.kl_coeff_optim.step()
                            self.kl_coeff_optim.zero_grad()
                    if self.ema_model is not None:
                        self.strategy.moving_average(
                            self.model, self.ema_model, self.ema_beta, torch.cuda.current_device()
                        )

                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()
                logs_dict = {
                    "loss": loss.item(),
                    "loss_mean": loss_mean,
                    "kl_estimate": kl_estimate.item(),
                    "max_value": max_value.item(),
                    "min_value": min_value.item(),
                }
                if clip_ratio is not None:
                    logs_dict.update({"clip_ratio": clip_ratio.item()})
                if train_actor and (self.single_step_actor_loss or self.dro_actor_loss):
                    actor_loss_mean = actor_loss_mean * 0.9 + 0.1 * actor_loss.item()
                    logs_dict.update({"actor_loss": actor_loss.item(), "actor_loss_mean": actor_loss_mean})
                if self.traj_actor_loss:
                    logs_dict.update({"vs0_max": vs0.max().item(), "vs0_min": vs0.min().item()})
                if train_actor and (self.kl_reg is not None or self.kl_targ is not None):
                    logs_dict.update({"kl_reg": kl_estimate_actor.item()})
                    if self.kl_targ is not None:
                        logs_dict.update({"log_kl_coeff": self.log_kl_coeff.item()})
                if self.plot_weights:
                    if weights is None:
                        with torch.no_grad():
                            weights, log_weights = self.weights(logps, reference_logps, action_masks)
                    for i in range(rewards.shape[0]):
                        if rewards[i].item() == 1:
                            pos_log_weights.append(log_weights[i].item())
                        else:
                            neg_log_weights.append(log_weights[i].item())
                    tmp = torch.tensor(
                        [len(pos_log_weights) > 0, len(neg_log_weights) > 0],
                        dtype=torch.long,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_reduce(tmp, torch.distributed.ReduceOp.MIN)
                    # if len(pos_log_weights) > 0 and len(neg_log_weights) > 0:
                    if tmp.bool().all().item():
                        pos_log_weights = pos_log_weights[-100:]
                        neg_log_weights = neg_log_weights[-100:]
                        logs_dict.update({"log_weights_recent_median_pos": np.median(pos_log_weights)})
                        logs_dict.update({"log_weights_recent_median_neg": np.median(neg_log_weights)})
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if False and global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def accumulated_logps(self, model, ids: torch.Tensor, masks: torch.Tensor, action_masks: torch.Tensor):
        outputs = model(ids, attention_mask=masks, return_output=True)
        logits = outputs["logits"]

        logits = logits[:, :-1, :]
        labels = ids[:, 1:]  # [bsz, seq_len]
        action_masks = action_masks[:, 1:]

        logps_raw = torch.log_softmax(logits, dim=-1)  # [bsz, seq_len, vocab]
        logps = logps_raw.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        accumulated_logps = (logps * action_masks).flip(-1).cumsum(-1).flip(-1)

        return accumulated_logps, logps, logps_raw

    # def forward(
    #     self,
    #     model,
    #     ids: torch.Tensor,
    #     masks: torch.Tensor,
    #     action_masks: torch.Tensor,
    #     critic=None,
    # ):
    #     outputs = model(ids, attention_mask=masks, return_output=True)
    #     logits = outputs["logits"]
    #     accumulated_logps = self.accumulated_logps(ids, logits, action_masks)
    #     if critic is not None:
    #         values = critic(ids, action_mask=action_masks, attention_mask=masks)
    #         return accumulated_logps, values
    #     return accumulated_logps

    def loss(
        self,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        rewards: torch.Tensor,
        clip_log: float | None = None,
        weights: torch.Tensor | None = None,
    ):
        policy_rewards = accumulated_logps - reference_accumulated_logps
        if clip_log is not None:
            clip_ratio = torch.mean((policy_rewards.abs() > clip_log).float())
            policy_rewards = policy_rewards.clip(-clip_log, clip_log)
        else:
            clip_ratio = None
        policy_rewards = self.beta * policy_rewards

        state_masks = state_masks[:, :-1]
        rewards = rewards.unsqueeze(1)
        if not self.importance_sampling_2:
            tmp = torch.square((policy_rewards + values - rewards) * state_masks).sum(dim=-1) / state_masks.sum(dim=-1)
            if weights is not None:
                tmp = tmp * weights
            return tmp.mean(), clip_ratio
        else:
            tmp = torch.square((policy_rewards + values - rewards) * state_masks)
            tmp = tmp * weights.unsqueeze(-1)
            return tmp.sum() / state_masks.sum(), clip_ratio

    def dro_loss(
        self,
        logps: torch.Tensor,
        reference_logps: torch.Tensor,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        rewards: torch.Tensor,
        weights: torch.Tensor | None = None,
    ):
        if not self.step_level:
            policy_rewards = accumulated_logps - reference_accumulated_logps
            policy_rewards = torch.cat([policy_rewards[:, 1:], torch.zeros_like(values[:, -1]).unsqueeze(-1)], dim=-1)
            policy_rewards = policy_rewards.detach() + logps - reference_logps
        else:
            next_value_indices = torch.zeros_like(values, dtype=torch.long)
            for i in range(values.shape[1] - 2, -1, -1):
                next_value_indices[:, i] = torch.where(state_masks[:, i + 1] == 1, i + 1, next_value_indices[:, i + 1])
            policy_rewards = accumulated_logps - reference_accumulated_logps
            tmp = torch.gather(policy_rewards, -1, next_value_indices)
            tmp = tmp.masked_fill(next_value_indices == 0, 0)
            policy_rewards = policy_rewards - tmp + tmp.detach()

        policy_rewards = self.beta * policy_rewards

        state_masks = state_masks[:, :-1]
        rewards = rewards.unsqueeze(1)
        if not self.importance_sampling_2:
            tmp = torch.square((policy_rewards + values - rewards) * state_masks).sum(dim=-1) / state_masks.sum(dim=-1)
            if weights is not None:
                tmp = tmp * weights
            return tmp.mean()
        else:
            tmp = torch.square((policy_rewards + values - rewards) * state_masks)
            tmp = tmp * weights.unsqueeze(-1)
            return tmp.sum() / state_masks.sum()

    def single_step_loss(
        self,
        logps: torch.Tensor,
        reference_logps: torch.Tensor,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        reward: torch.Tensor,
    ):
        # TODO: only supports token-level MDP for now
        # TODO: only supports left-padding for now
        if self.step_level:
            next_value_indices = torch.zeros_like(values, dtype=torch.long)
            for i in range(values.shape[1] - 2, -1, -1):
                next_value_indices[:, i] = torch.where(state_masks[:, i + 1] == 1, i + 1, next_value_indices[:, i + 1])
            next_values = torch.gather(values, -1, next_value_indices)
            next_values = next_values.masked_fill(next_value_indices == 0, 0)
            tmp = torch.gather(accumulated_logps - reference_accumulated_logps, -1, next_value_indices)
            tmp = tmp.masked_fill(next_value_indices == 0, 0)
            policy_rewards = self.beta * (accumulated_logps - reference_accumulated_logps - tmp)
        else:
            next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1]).unsqueeze(-1)], dim=-1)
            policy_rewards = self.beta * (logps - reference_logps)
        rewards = torch.zeros_like(values)
        for i in range(state_masks.shape[0]):
            j = state_masks[i].nonzero()[-1].item()
            rewards[i, j] = reward[i]

        state_masks = state_masks[:, :-1]
        tmp = torch.square((values - rewards - next_values + policy_rewards) * state_masks).sum(
            dim=-1
        ) / state_masks.sum(dim=-1)
        return tmp.mean()

    def traj_loss(
        self,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        rewards: torch.Tensor,
    ):
        policy_rewards = accumulated_logps - reference_accumulated_logps
        policy_rewards = self.beta * policy_rewards

        indices = []
        bsz = accumulated_logps.shape[0]
        for i in range(bsz):
            indices.append(state_masks[i].nonzero()[0].item())
        indices = torch.tensor(indices, dtype=torch.long, device=accumulated_logps.device)

        state_masks = state_masks[:, :-1]
        rewards = rewards.unsqueeze(1)
        tmp = torch.square((policy_rewards + values - rewards) * state_masks)
        tmp = torch.gather(tmp, -1, indices.unsqueeze(-1))
        vs0 = torch.gather(values, -1, indices.unsqueeze(-1))
        loss = tmp.mean()
        if self.hinge_coeff is not None:
            hinge_loss = self.hinge_coeff * (reference_accumulated_logps - accumulated_logps).clamp(min=0)
            hinge_loss = torch.gather(hinge_loss, -1, indices.unsqueeze(-1))
            pos_mask = rewards == 1
            if pos_mask.any():
                loss = loss + hinge_loss.masked_fill(~pos_mask, 0).sum() / pos_mask.sum()
        return loss, vs0

    def weights(self, logps: torch.Tensor, reference_logps: torch.Tensor, action_masks: torch.Tensor):
        tmp = torch.masked_fill(logps - reference_logps, ~action_masks[:, 1:].bool(), 0).sum(dim=-1)
        return tmp.clip(max=1).exp(), tmp
