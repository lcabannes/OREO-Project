set -x

printf -v INPUT_TEMPLATE "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response: Let's think step by step.\n"

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 8192 \
    --dataset openai/gsm8k \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 32 \
    --pretrain /mnt/minicpm \
    --save_path ./ckpt/minicpm_sft_gsm8k \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 10 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 5e-6 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --input_key question \
    --output_key answer \
    --use_wandb True \
    --wandb_org jwhj \
    --wandb_project pcl
EOF

INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed --include $INCLUDE $training_commands --input_template "$INPUT_TEMPLATE"
fi