set -x 

read -r -d '' training_commands <<EOF
../train_pcl.py \
     --save_path /mnt/lustre/ckpt/dsm-inst_oreo \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 2 \
     --pretrain deepseek-ai/deepseek-math-7b-instruct \
     --bf16 \
     --max_epochs 2 \
     --max_len 2048 \
     --zero_stage 2 \
     --beta 0.03 \
     --learning_rate 5e-6 \
     --critic_learning_rate 1e-4 \
     --lora_rank 64 \
     --lora_alpha 64 \
     --only_critic_lora \
     --adam_offload \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload \
     --dro_actor_loss \
     --rew_mul 1 \
     --rew_add 0 \
     --kl_reg 0.01 \
     --unbiased_kl \
     --plot_weights \
     --fix_max_steps \
     --padding_side left \
     --train_file /mnt/shared/annotated/train-dsm-b2.jsonl
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    deepspeed --include $INCLUDE $training_commands
fi
