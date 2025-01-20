import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel
import torch
from openrlhf.models import Actor, get_llm_for_sequence_regression

def main():
    if args.is_critic:
        print(f"loading critic")
        critic = get_llm_for_sequence_regression(
            args.base_model, 
            "critic",
            zero_init_value_head=True,
            bf16=True,
        )
        model = PeftModel.from_pretrained(critic, args.lora_path)

    elif args.base_model != "":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(args.lora_path, trust_remote_code=True)
    
    merged = model.merge_and_unload()
    dirname, filename = os.path.split(args.lora_path)
    dest = os.path.join(dirname, filename + "_merged")
    print(dest)

    merged.save_pretrained(dest)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    tokenizer.save_pretrained(dest)
    print(f"model and tokenizer saved to {dest}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--is_critic", action="store_true")
    args = parser.parse_args()

    main()
