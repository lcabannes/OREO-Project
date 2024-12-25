NAME=dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-dro_actor-freeze-ratio0-2
python ../merge_lora.py --lora_path /mnt/lustre/ckpt/$NAME
python ../run_dsm.py --model /mnt/lustre/ckpt/${NAME}_merged --no_bos --save /mnt/shared/annotated/test-${NAME}_merged.jsonl