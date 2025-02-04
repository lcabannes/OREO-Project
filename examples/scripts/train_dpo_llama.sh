set -x 

read -r -d '' training_commands <<EOF
/home/OREO-Project/examples/train_dpo.py \
     --save_path /home/OREO-Project/data/ckpt/SmolLM-dpo/ \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 4 \
     --pretrain HuggingFaceTB/SmolLM-360M-Instruct \
     --bf16 \
     --max_epochs 2 \
     --max_len 2560 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,lmsys/chatbot_arena_conversations,openai/webgpt_comparisons \
     --dataset_probs 0.72,0.08,0.12,0.08 \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
