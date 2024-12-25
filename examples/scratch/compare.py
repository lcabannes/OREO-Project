import json

exp1 = "dev-mistral_alfworld_pcl_b_kl-reg100_2epochs_merged"
exp2 = "dev-eto-mistral-alfworld-sft"


for i in range(140):
    with open(f"/mnt/shared/annotated/{exp1}/{i}.json", "r") as f:
        t1 = json.load(f)
    with open(f"/mnt/shared/annotated/{exp2}/{i}.json", "r") as f:
        t2 = json.load(f)
    if t1["meta"]["reward"] != t2["meta"]["reward"]:
        print(i)
