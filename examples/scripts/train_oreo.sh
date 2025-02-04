set -x 

read -r -d '' training_commands <<EOF
/home/OREO-Project/examples/train_pcl.py \
     --save_path /home/OREO-Project/data/ckpt/Qwen-math-oreo/ \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain Qwen/Qwen2.5-Math-1.5B-Instruct \
     --bf16 \
     --max_epochs 2 \
     --ref_offload \
     --max_len 1750 \
     --zero_stage 2 \
     --beta 0.03 \
     --learning_rate 5e-6 \
     --critic_learning_rate 5e-6 \
     --adam_offload \
     --lora_rank 124 \
     --flash_attn \
     --gradient_checkpointing \
     --dro_actor_loss \
     --rew_mul 1 \
     --rew_add 0 \
     --kl_reg 0.01 \
     --unbiased_kl \
     --plot_weights \
     --padding_side left \
     --train_file /home/OREO-Project/train-qwen.jsonl \
     --use_wandb True \
     --wandb_project oreo \
     --wandb_run_name Qwen2.5-Math-1.5B \

EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

# 
     # --pretrain /home/OREO-Project/data/ckpt/models--meta-llama--Llama-3.2-1B-instruct/snaspshots/files/ \
INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
echo $INCLUDE
if [[ ${1} != "slurm" ]]; then
    deepspeed --include $INCLUDE $training_commands
fi
