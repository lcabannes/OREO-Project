import os
import json
import re

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

task_prompt = [
    {
        "role": "user",
        "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
    },
    {"role": "assistant", "content": "OK"},
]


def sample_pairs(tokenizer: PreTrainedTokenizer, pos_resp, neg_resp):
    if len(pos_resp) == 0 or len(neg_resp) == 0:
        return []
    indices = []
    for i in range(len(pos_resp)):
        for j in range(len(neg_resp)):
            indices.append((i, j))
    tmp = np.random.choice(range(len(indices)), min(5, len(indices)), replace=False)
    indices = [indices[i] for i in tmp]
    results = []
    for i, j in indices:
        prompt = task_prompt + [pos_resp[i][2]]
        chosen = pos_resp[i][3:]
        rejected = neg_resp[j][3:]
        results.append(
            {
                "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
                "chosen": tokenizer.apply_chat_template(chosen, tokenize=False),
                "rejected": tokenizer.apply_chat_template(rejected, tokenize=False),
            }
        )
    return results


path = "/mnt/shared/annotated/annotated/train-n5-t1-minicpm_alfworld_sft_full/"
r = re.compile("^(.*)_(.*).json$")
indices = {}
for file in os.listdir(path):
    file: str
    if not file.endswith(".json"):
        continue
    with open(os.path.join(path, file), "r") as f:
        traj_raw = json.load(f)
    tmp = r.match(file)
    assert tmp is not None
    task_id = tmp.group(1)
    repeat_id = tmp.group(2)
    if task_id not in indices:
        indices[task_id] = [[], []]
    traj: list = traj_raw["conversations"][2:]
    if traj[-1]["role"] == "user":
        traj.pop()
    if len(traj) == 0:
        __import__("pdb").set_trace()
    if traj_raw["meta"]["reward"]:
        indices[task_id][0].append(traj)
    else:
        indices[task_id][1].append(traj)

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/data/ckpt/pcl/minicpm_full_alfworld_pcl_token-level_beta0-03_actor-loss-dro_kl-reg-unbiased1e-2_plot-weights_3epochsf",
    trust_remote_code=True,
)

paired_data = []
for task_id in indices.keys():
    paired_data.extend(sample_pairs(tokenizer, indices[task_id][0], indices[task_id][1]))

with open("/mnt/shared/annotated/train-alfworld-dpo.jsonl", "w") as f:
    for item in paired_data:
        f.write(json.dumps(item))
        f.write("\n")
