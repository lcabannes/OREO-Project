from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from openrlhf.datasets.answer_extraction import extract_last_single_answer, extract_math_answer
from openrlhf.datasets.eval.eval_script import eval_last_single_answer, eval_math

a = LLM("deepseek-ai/deepseek-math-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
params = SamplingParams(n=20, temperature=1, max_tokens=2048)

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# q = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
q = "Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction."
s: str = tokenizer.apply_chat_template(
    [{"role": "user", "content": q + template}], add_generation_prompt=True, tokenize=False
)

result = a.generate(s, sampling_params=params)[0]
for output in result.outputs:
    pred = extract_math_answer(q, output.text, "")
    print(pred)
    reward = eval_math({"prediction": pred, "answer": ["\\dfrac{9}{7}"]})
    print(reward)
