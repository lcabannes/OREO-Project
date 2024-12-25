from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from openrlhf.datasets.pcl_dataset_alfworld import PCLDatasetAlfWorld

tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/eto-mistral-alfworld-sft")
dataset = PCLDatasetAlfWorld(
    tokenizer, 8192, train_file="/mnt/shared/annotated/train-alfworld.json", step_level=True, padding_side="left"
)

tmp = []
for i in tqdm(range(len(dataset))):
    ids = dataset[i][0]
    tmp.append(ids.shape[1])

plt.hist(tmp, bins=40)
plt.savefig("alfworld_length.jpg")
