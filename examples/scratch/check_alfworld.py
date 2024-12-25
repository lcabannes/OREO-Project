import yaml
import json
import os

import alfworld.agents.environment as environment

with open(os.path.join("/mnt/shared/alfworld/configs", "base_config.yaml")) as f:
    config = yaml.safe_load(f)

env_type = "AlfredTWEnv"
env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)
env.skip(88)

with open("/mnt/shared/annotated/test-mistral_alfworld_pcl_kl-reg100_2epochs_merged/88.json", "r") as f:
    traj_raw = json.load(f)

obs, info = env.reset()

for i in range(3, len(traj_raw["conversations"]), 2):
    action_text = traj_raw["conversations"][i]["content"]
    action = action_text.split("Action: ")[1].strip()

    obs, scores, dones, infos = env.step([action])
    ob = obs[0]
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]

    assert "Observation: " + ob == traj_raw["conversations"][i + 1]["content"]
    print(scores)
