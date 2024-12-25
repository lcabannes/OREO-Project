NAME=qwen_full_lr5e-6_beta0-03_actor-loss-dro_kl-reg-unbiased3e-2_plot-weights
python ../scratch/run_qwen.py --model /mnt/data/ckpt/${NAME} --save /mnt/shared/annotated/test-${NAME}.jsonl