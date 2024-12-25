import os
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def main():
    model = AutoPeftModelForCausalLM.from_pretrained(args.lora_path, trust_remote_code=True)
    merged = model.merge_and_unload()
    dirname, filename = os.path.split(args.lora_path)
    dest = os.path.join(dirname, filename + "_merged")
    print(dest)

    merged.save_pretrained(dest)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    tokenizer.save_pretrained(dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    args = parser.parse_args()

    main()
