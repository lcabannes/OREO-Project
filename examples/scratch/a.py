import json

from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def length_dist(filename: str, model_uri: str, label=None):
    data = []
    Ls = []
    tokenizer = AutoTokenizer.from_pretrained(model_uri)

    with open(filename, "r") as f:
        for line in tqdm(f.readlines()):
            item = json.loads(line)
            data.append(item)
            inputs = tokenizer(item["prompt"] + item["response"] + " " + tokenizer.eos_token)
            Ls.append(len(inputs.input_ids))

    plt.hist(Ls, density=True, bins=100, label=label)


length_dist("/mnt/shared/annotated/test-a_lr1e-4_actor-lr5e-5_merged.jsonl", "/mnt/minicpm", "minicpm SFT -> PCL")
length_dist(
    "/mnt/shared/annotated/test-mistral_lr1e-4_actor-lr5e-5_wrong-dataset_merged.jsonl",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral SFT -> PCL (minicpm dataset)",
)
length_dist(
    "/mnt/shared/annotated/test-mistral_lr1e-4_actor-lr5e-5_merged.jsonl",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral SFT -> PCL",
)
plt.legend()
plt.savefig("a.jpg")

plt.close()
length_dist("/mnt/shared/annotated/train-n5-t1.jsonl", "/mnt/minicpm", "minicpm SFT (train)")
length_dist(
    "/mnt/shared/annotated/train-mistral-n5-t1.jsonl", "mistralai/Mistral-7B-Instruct-v0.2", "mistral SFT (train)"
)
plt.legend()
plt.savefig("b.jpg")
