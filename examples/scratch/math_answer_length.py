from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
d = load_dataset("hendrycks/competition_math")

Ls = []
for item in tqdm(d["train"]):
    answer = item["solution"]
    L = len(tokenizer(answer)["input_ids"])
    Ls.append(L)

plt.hist(Ls, bins=100)
plt.savefig("math_answer_length.jpg")
