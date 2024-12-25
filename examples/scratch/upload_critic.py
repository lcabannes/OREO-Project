from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression

model_path = "/mnt/data/ckpt/pcl/qwen_full_lr5e-6_beta0-03_rew01_actor-loss-dro_kl-reg-unbiased1e-2_plot-weights"
critic = get_llm_for_sequence_regression(
    model_path + "_critic",
    "critic",
    normalize_reward=False,  # TODO: maybe experiment with this layer
    use_flash_attention_2=True,
    bf16=True,
    # load_in_4bit=False,
    # lora_rank=64,
    # lora_alpha=64,
    # lora_dropout=0,
    # target_modules="all-linear",
    # ds_config=strategy.get_ds_train_config(is_actor=True),
)

critic.push_to_hub("jwhj/Qwen2.5-Math-1.5B-PCL-Value")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.push_to_hub("jwhj/Qwen2.5-Math-1.5B-PCL-Value")
