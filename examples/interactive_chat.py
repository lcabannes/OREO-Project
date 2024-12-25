import argparse
import torch
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer


def generate(args):
    # dummy strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
    )

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model.model, "left", dummy_strategy, use_fast=not args.disable_fast_tokenizer
    )

    if args.ta_prompt:
        with open(args.ta_prompt, "r") as f:
            user_prompt = f.read()
    else:
        user_prompt = args.system_prompt

    while True:
        inputs = input("Please enter a prompt (or type 'exit' to quit): ")
        if inputs.strip().lower() == "exit":
            print("Exiting program...")
            break
        if inputs.strip().lower() == "clear":
            user_prompt = args.system_prompt
            continue

        # get input prompt
        user_prompt = user_prompt + "\n" + args.input_template.format(inputs)
        if args.enable_ca:
            user_prompt += args.ca_prompt.strip() + " "
        user_prompt_len = len(user_prompt)

        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(torch.cuda.current_device())
        outputs = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_length=args.max_len,
            do_sample=not args.greedy_sampling,
            top_p=args.top_p,
            early_stopping=True,
            num_beams=1,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        user_prompt = output[0]
        output = output[0][user_prompt_len:].replace(r"\n", "\n")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--system_prompt", type=str, default="")

    # QLora
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument("--enable_ca", action="store_true", default=False)
    parser.add_argument("--ca_prompt", type=str, default="<rm_score>: 5.00", help="conditional SFT prompt")
    args = parser.parse_args()

    print(args)
    generate(args)
