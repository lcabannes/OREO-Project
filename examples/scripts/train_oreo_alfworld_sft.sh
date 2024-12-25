set -x

printf -v INPUT_TEMPLATE "{}"

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 8192 \
    --dataset /mnt/shared/annotated/minicpm_alfworld_sft.jsonl \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 4 \
    --pretrain openbmb/MiniCPM-2B-dpo-bf16 \
    --save_path /mnt/lustre/ckpt/minicpm_alfworld_sft_full \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps 250 \
    --zero_stage 2 \
    --max_epochs 2 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 2e-5 \
    --input_key prompt \
    --output_key response \
    --padding_side left
EOF

INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed --include $INCLUDE $training_commands --input_template "$INPUT_TEMPLATE"
fi