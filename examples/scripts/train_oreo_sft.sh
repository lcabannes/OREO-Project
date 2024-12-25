set -x

printf -v INPUT_TEMPLATE "{}"

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 8192 \
    --dataset /mnt/shared/annotated/train-qwen-sft.jsonl \
    --dataset_probs 1.0 \
    --padding_side left \
    --train_batch_size 128 \
    --micro_train_batch_size 8 \
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --save_path /mnt/data/ckpt/qwen_sft \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps 100 \
    --zero_stage 2 \
    --max_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 2e-5 \
    --input_key prompt \
    --output_key response
EOF

INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed --include $INCLUDE $training_commands --input_template "$INPUT_TEMPLATE"
fi