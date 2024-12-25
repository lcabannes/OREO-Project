import argparse
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import json
import yaml

from tqdm import tqdm
import openai
from openai.types.chat.chat_completion import ChatCompletion
from transformers import PreTrainedTokenizer
import alfworld.agents.environment as environment

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

icl = [
    {
        "role": "user",
        "content": "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger.",
    },
    {
        "role": "assistant",
        "content": "Thought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1",
    },
    {"role": "user", "content": "Observation: On the toiletpaperhanger 1, you see nothing."},
    {
        "role": "assistant",
        "content": "Thought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1",
    },
    {"role": "user", "content": "Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1."},
    {
        "role": "assistant",
        "content": "Thought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1",
    },
    {"role": "user", "content": "Observation: You pick up the toiletpaper 1 from the toilet 1."},
    {
        "role": "assistant",
        "content": "Thought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1",
    },
    {"role": "user", "content": "Observation: On the toiletpaperhanger 1, you see nothing."},
    {
        "role": "assistant",
        "content": "Thought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1",
    },
]


def run(start: int, end: int, prompt: list, save_dir: str, repeat_id: int | None = None):
    # load config
    with open(os.path.join("/mnt/shared/alfworld/configs", "base_config.yaml")) as f:
        config = yaml.safe_load(f)
    env_type = "AlfredTWEnv"
    env = getattr(environment, env_type)(config, train_eval=args.split)
    env = env.init_env(batch_size=1)
    env.skip(start)

    for env_id in tqdm(range(start, end)):
        conversation = deepcopy(prompt)
        obs, info = env.reset()
        conversation.append({"role": "user", "content": obs[0][len("-= Welcome to TextWorld, ALFRED! =-\n\n") :]})

        reward = 0
        invalid = False
        for _ in range(40):
            try:
                resp: ChatCompletion = client.chat.completions.create(
                    messages=conversation,
                    model=args.model,
                    max_tokens=2048,
                    temperature=args.temperature,
                    stop="<用户>",
                )
            except:
                invalid = True
                break
            action_text = resp.choices[0].message.content.strip()

            conversation.append({"role": "assistant", "content": action_text})
            try:
                action = action_text.split("Action: ")[1]
            except:
                invalid = True
                break

            obs, scores, dones, infos = env.step([action])
            ob = obs[0]
            if ob.startswith("You arrive at loc "):
                ob = ob[ob.find(". ") + 2 :]
            conversation.append({"role": "user", "content": "Observation: " + ob})

            if dones[0]:
                reward = scores[0]
                break

        traj = {"meta": {"reward": bool(reward), "invalid": invalid}, "conversations": conversation}
        suffix = "" if repeat_id is None else f"_{repeat_id}"
        with open(os.path.join(save_dir, f"{env_id}{suffix}.json"), "w") as f:
            json.dump(traj, f, indent=4)

    # return traj


def main():
    chunksize = 16

    # # load config
    # with open(os.path.join("/mnt/shared/alfworld/configs", "base_config.yaml")) as f:
    #     config = yaml.safe_load(f)
    # # env_type = config["env"]["type"]  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    # env_type = "AlfredTWEnv"
    # N_TASKS = 134
    # part_inst_num = [N_TASKS // chunksize] * chunksize
    # for i in range(N_TASKS % chunksize):
    #     part_inst_num[i] += 1
    # for i in range(chunksize):
    #     env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
    #     env = env.init_env(batch_size=1)
    #     env.skip(sum(part_inst_num[:i]))
    #     q.put((env, sum(part_inst_num[:i]), part_inst_num[i]))

    if args.split == "train":
        N_TASKS = 3321
    elif args.split == "eval_in_distribution":
        N_TASKS = 140
    elif args.split == "eval_out_of_distribution":
        N_TASKS = 134
    else:
        raise "unknown split"
    part_inst_num = [N_TASKS // chunksize] * chunksize
    for i in range(N_TASKS % chunksize):
        part_inst_num[i] += 1
    ranges_l = []
    ranges_r = []
    offset = 0
    for i in range(chunksize):
        ranges_l.append(offset)
        ranges_r.append(offset + part_inst_num[i])
        offset += part_inst_num[i]
    if args.repeat is not None:
        ranges_l = [x for x in ranges_l for _ in range(args.repeat)]
        ranges_r = [x for x in ranges_r for _ in range(args.repeat)]
        repeat_ids = list(itertools.chain.from_iterable(range(args.repeat) for _ in range(chunksize)))
        assert len(ranges_l) == len(ranges_r) and len(ranges_l) == len(repeat_ids)
    else:
        repeat_ids = itertools.repeat(None)

    prompt = [
        {
            "role": "user",
            "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
        },
        {"role": "assistant", "content": "OK"},
    ]

    with ProcessPoolExecutor(chunksize) as executor:
        # with ThreadPoolExecutor(chunksize) as executor:
        jobs = executor.map(
            run, ranges_l, ranges_r, itertools.repeat(prompt), itertools.repeat(args.save_dir), repeat_ids
        )
        _ = list(jobs)

    # with open("run_async_results.json", "w") as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main()
