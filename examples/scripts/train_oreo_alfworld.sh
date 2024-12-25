set -x 

read -r -d '' training_commands <<EOF
../train_pcl.py \
     --save_path /mnt/lustre/ckpt/minicpm_oreo_alfworld \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 2 \
     --pretrain /mnt/lustre/ckpt/minicpm_alfworld_sft_full \
     --bf16 \
     --max_epochs 5 \
     --max_len 4096 \
     --zero_stage 2 \
     --beta 0.03 \
     --learning_rate 5e-6 \
     --critic_learning_rate 5e-6 \
     --adam_offload \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload \
     --task alfworld \
     --dro_actor_loss \
     --step_level \
     --token_level_agent \
     --kl_reg 0.01 \
     --unbiased_kl \
     --plot_weights \
     --fix_max_steps \
     --padding_side left \
     --train_file /mnt/shared/annotated/train-n5-t1-minicpm-alfworld.json
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    deepspeed --include $INCLUDE $training_commands
fi
