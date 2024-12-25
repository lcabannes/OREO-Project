import argparse
import os
import json

task_prompt = [
    {
        "role": "user",
        "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
    },
    {"role": "assistant", "content": "OK"},
]


def main():
    data = []
    success = 0
    total = 0
    for trajs_path in args.trajs_path:
        for file in os.listdir(trajs_path):
            file: str
            if not file.endswith(".json"):
                continue
            with open(os.path.join(trajs_path, file), "r") as f:
                traj_raw = json.load(f)
            if args.no_icl:
                offset = 2
            else:
                offset = 12
            traj: list = traj_raw["conversations"][offset:]

            if not args.is_chat_style:
                for i in range(len(traj)):
                    traj[i] = {
                        "role": "user" if traj[i]["from"] == "human" else "assistant",
                        "content": traj[i]["value"],
                    }

            if traj[-1]["role"] == "user":
                traj.pop()
            if len(traj) == 0:
                __import__("pdb").set_trace()
            datum = {"conversations": task_prompt + traj, "reward": 1 if traj_raw["meta"]["reward"] else 0}
            if not (args.failed_only and traj_raw["meta"]["reward"]):
                data.append(datum)
            total += 1
            success += datum["reward"]
    print("{} / {}".format(success, total))

    if args.golden_trajs is not None:
        with open(os.path.join(args.golden_trajs), "r") as f:
            golden_trajs_raw = json.load(f)
        for item in golden_trajs_raw:
            traj: list = item["conversations"]
            for i in range(len(traj)):
                traj[i] = {"role": "user" if traj[i]["from"] == "human" else "assistant", "content": traj[i]["value"]}
            data.append({"conversations": traj, "reward": 1})
        print(len(golden_trajs_raw))

    with open(args.save, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_trajs", type=str)
    parser.add_argument("--trajs_path", nargs="+", required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--is_chat_style", action="store_true")
    parser.add_argument("--no_icl", action="store_true")
    parser.add_argument("--failed_only", action="store_true")
    args = parser.parse_args()

    main()
