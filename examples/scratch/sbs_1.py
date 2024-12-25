import json

from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

extract_answer = extract_last_single_answer
eval_answer = eval_last_single_answer

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
critic = get_llm_for_sequence_regression(
    "deepseek-ai/deepseek-math-7b-instruct",
    "critic",
    normalize_reward=False,  # TODO: maybe experiment with this layer
    use_flash_attention_2=True,
    bf16=True,
    load_in_4bit=False,
    lora_rank=64,
    lora_alpha=64,
    lora_dropout=0,
    target_modules="all-linear",
    # ds_config=strategy.get_ds_train_config(is_actor=True),
    zero_init_value_head=True,
)
critic.to("cuda")

critic.load_adapter("/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_critic", "default")

a = LLM(
    "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_merged", gpu_memory_utilization=0.5
)
params = SamplingParams(temperature=1, stop="\n", max_tokens=2048)

d = load_dataset("openai/gsm8k", "main")["test"]
# d = load_dataset("hendrycks/competition_math")

logs = []
for problem_idx in tqdm(range(10), desc="Solving"):
    question = d[problem_idx]["question"]
    gt_answer = d[problem_idx]["answer"]

    states = [chat_template.format(question + template)]
    answers = []
    K = 3
    for _ in range(16):
        next_states = []
        for repeat_id in range(K):
            results = a.generate(states, params, use_tqdm=False)
            completions = [result.outputs[0].text for result in results]
            input_token = tokenizer(
                [state + completion for state, completion in zip(states, completions)],
                padding=True,
                return_tensors="pt",
            )
            input_token.to("cuda")
            with torch.no_grad():
                values = critic(**input_token, action_mask=input_token["input_ids"])
            for i in range(len(completions)):
                next_states.append(
                    (states[i] + completions[i] + "\n", values[i, -1].item(), results[i].outputs[0].stop_reason)
                )
        next_states = sorted(next_states, key=lambda x: -x[1])[:K]
        states = []
        for state in next_states:
            if state[2] is None:
                answers.append(state)
            else:
                states.append(state[0])
        if len(states) == 0:
            break

    answers = sorted(answers, key=lambda x: -x[1])
    prompt = chat_template.format(question + template)
    pred = extract_answer(question, answers[0][0][len(prompt) :], "")

    answer = extract_answer(question, gt_answer, "")

    reward = int(eval_answer({"prediction": pred, "answer": answer}))
    logs.append({"prompt": prompt, "response": answers[0][0][len(prompt) :], "reward": reward})

with open("sbs.jsonl", "w") as f:
    for log in logs:
        f.write(json.dumps(log))
        f.write("\n")
