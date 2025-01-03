import argparse
import math
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import torch

from transformers.trainer import get_scheduler

from openrlhf.datasets.pcl_dataset import PCLDataset
from openrlhf.datasets.pcl_dataset_alfworld import PCLDatasetAlfWorld
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer.pcl_trainer import PCLTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    # TODO: initialize a separate actor and critic;
    # You may refer to the implemetation in train_ppo.py
    # model = ActorCritic(
    #     args.pretrain,
    #     use_flash_attention_2=args.flash_attn,
    #     bf16=args.bf16,
    #     load_in_4bit=args.load_in_4bit,
    #     lora_rank=args.lora_rank,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     target_modules=args.target_modules,
    #     ds_config=strategy.get_ds_train_config(is_actor=True),
    # )
    print(f"only critic: {args.only_critic_lora}")
    print(f"lora rank: {args.lora_rank}")
    model = Actor(
        args.load_actor,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank if not args.only_critic_lora else 0,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        device_map="cpu" # lcabannes added this
    )
    torch.cuda.synchronize()
    print(f"LOADING CRITIC")
    critic = get_llm_for_sequence_regression(
        args.pretrain if args.load_critic is None else args.load_critic,
        "critic",
        normalize_reward=False,  # TODO: maybe experiment with this later
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        zero_init_value_head=args.load_critic is None,
        device_map="cpu",
    )
    torch.cuda.synchronize()
    print(f"CRITIC LOADED")
    import time
    time.sleep(3)
    if args.load_critic_adapter is not None:
        critic.load_adapter(args.load_critic_adapter, "default")
    if args.enable_ema:
        assert args.lora_rank == 0 or args.only_critic_lora
        ema_model = Actor(
            args.load_actor,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model.model, args.padding_side, strategy, use_fast=not args.disable_fast_tokenizer
    )
    get_tokenizer(args.pretrain, critic, args.padding_side, strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)
    strategy.print(critic)

    # load weights for ref model
    ref_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        # device_map="cuda",
    )
    if args.ref_offload:
        ref_model._offload = True
    get_tokenizer(
        args.pretrain, ref_model.model, args.padding_side, strategy, use_fast=not args.disable_fast_tokenizer
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )

    # prepare for data and dataset
    # TODO:
    dataset_cls = PCLDataset if args.task == "math" else PCLDatasetAlfWorld
    step_level = args.step_level and not args.token_level_agent
    train_dataset = dataset_cls(
        tokenizer,
        args.max_len,
        train_file=args.train_file,
        step_level=step_level,
        period_as_delimiter=args.period_as_delimiter,
        padding_side=args.padding_side,
        rew_mul=args.rew_mul,
        rew_add=args.rew_add,
    )
    eval_dataset = dataset_cls(
        tokenizer,
        args.max_len,
        train_file=args.train_file,
        split="test",
        step_level=step_level,
        period_as_delimiter=args.period_as_delimiter,
        padding_side=args.padding_side,
        rew_mul=args.rew_mul,
        rew_add=args.rew_add,
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # scheduler
    if args.fix_max_steps:
        num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    else:
        num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    actor_freeze_steps = (
        math.ceil(max_steps * args.actor_freeze_ratio) if args.actor_freeze_ratio is not None else None
    )

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps if actor_freeze_steps is None else max_steps - actor_freeze_steps,
    )

    critic_scheduler = get_scheduler(
        "cosine", critic_optim, num_warmup_steps=math.ceil(max_steps * 0.03), num_training_steps=max_steps
    )

    # strategy prepare
    (model, optim, scheduler), (critic, critic_optim, critic_scheduler), ref_model = strategy.prepare(
        (model, optim, scheduler), (critic, critic_optim, critic_scheduler), ref_model
    )
    # TODO: there is a parameter "is_rlhf" in PPO impl, figure out what it does
    if ema_model is not None:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model)

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = PCLTrainer(
        model=model,
        critic=critic,
        ref_model=ref_model,
        ema_model=ema_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        critic_optim=critic_optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        critic_scheduler=critic_scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
        critic_only=args.critic_only,
        step_level=args.step_level,
        single_step_actor_loss=args.single_step_actor_loss,
        dro_actor_loss=args.dro_actor_loss,
        traj_actor_loss=args.traj_actor_loss,
        clip_log=args.clip_log,
        ema_beta=args.ema_beta,
        kl_reg=args.kl_reg,
        kl_targ=args.kl_targ,
        init_log_kl_coeff=args.init_log_kl_coeff,
        kl_coeff_lr=args.kl_coeff_lr,
        unbiased_kl=args.unbiased_kl,
        forward_kl=args.forward_kl,
        hinge_coeff=args.hinge_coeff,
        actor_freeze_steps=actor_freeze_steps,
        importance_sampling=args.importance_sampling,
        importance_sampling_2=args.importance_sampling_2,
        plot_weights=args.plot_weights,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)
    strategy.save_model(critic, tokenizer, args.save_path + "_critic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    # parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    # parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--padding_side", type=str, default="right")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--load_actor", type=str)
    parser.add_argument("--load_critic", type=str)
    parser.add_argument("--load_critic_adapter", type=str)
    parser.add_argument("--step_level", action="store_true")
    parser.add_argument("--period_as_delimiter", action="store_true")
    parser.add_argument("--token_level_agent", action="store_true")
    parser.add_argument("--single_step_actor_loss", action="store_true")
    parser.add_argument("--dro_actor_loss", action="store_true")
    parser.add_argument("--traj_actor_loss", action="store_true")
    parser.add_argument("--clip_log", type=float)
    parser.add_argument("--critic_only", action="store_true")
    parser.add_argument("--only_critic_lora", action="store_true")
    parser.add_argument("--enable_ema", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--kl_reg", type=float)
    parser.add_argument("--kl_targ", type=float)
    parser.add_argument("--init_log_kl_coeff", type=float, default=0.0)
    parser.add_argument("--kl_coeff_lr", type=float, default=1e-3)
    parser.add_argument("--unbiased_kl", action="store_true")
    parser.add_argument("--forward_kl", action="store_true")
    parser.add_argument("--hinge_coeff", type=float)
    parser.add_argument("--actor_freeze_ratio", type=float)
    parser.add_argument("--importance_sampling", action="store_true")
    parser.add_argument("--importance_sampling_2", action="store_true")
    parser.add_argument("--rew_mul", type=float, default=2)
    parser.add_argument("--rew_add", type=float, default=-1)
    parser.add_argument("--plot_weights", action="store_true")
    parser.add_argument("--fix_max_steps", action="store_true")

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    if args.load_actor is None:
        args.load_actor = args.pretrain
    assert not (args.single_step_actor_loss and args.padding_side == "right")
    assert not (args.single_step_actor_loss and args.traj_actor_loss)
    if args.importance_sampling_2:
        assert args.importance_sampling

    train(args)
