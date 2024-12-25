from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

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

# tokenizer = AutoTokenizer.from_pretrained("/mnt/data/ckpt/minicpm_alfworld_sft", trust_remote_code=True)
# a = LLM("/mnt/data/ckpt/minicpm_alfworld_sft_merged", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/mistral_alfworld_sft_2", trust_remote_code=True)
# a = LLM("/mnt/lustre/ckpt/mistral_alfworld_sft_2_merged", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/eto-mistral-alfworld-sft", trust_remote_code=True)
# a = LLM("/mnt/lustre/ckpt/eto-mistral-alfworld-sft", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/mistral_alfworld_sft_full", trust_remote_code=True)
# a = LLM("/mnt/lustre/ckpt/mistral_alfworld_sft_full", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/ckpt/mistral_alfworld_pcl_merged", trust_remote_code=True)
# a = LLM("/mnt/lustre/ckpt/mistral_alfworld_pcl_merged", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(
#     "/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg10_merged", trust_remote_code=True
# )
# a = LLM("/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg10_merged", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(
#     "/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg100_merged", trust_remote_code=True
# )
# a = LLM("/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg100_merged", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg100_2epochs_merged", trust_remote_code=True
)
a = LLM("/mnt/lustre/ckpt/mistral_alfworld_pcl_kl-reg100_2epochs_merged", trust_remote_code=True)
params = SamplingParams(max_tokens=2048)

# load config
config = generic.load_config()
# env_type = config["env"]["type"]  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
env_type = "AlfredTWEnv"

# setup environment
# env = getattr(environment, env_type)(config, train_eval="eval_in_distribution")
env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)

# interact


def gen_two_line(a: LLM, prompt: str):
    result = a.generate(prompt, SamplingParams(max_tokens=2048, stop=["\n"]))
    text1 = result[0].outputs[0].text
    result = a.generate(prompt + text1.strip() + "\n", SamplingParams(max_tokens=2048, stop=["\n"]))
    text2 = result[0].outputs[0].text
    return text1 + "\n" + text2


success = 0
invalid = 0
with tqdm(range(134)) as pbar:
    for i in pbar:
        obs, info = env.reset()
        print(obs[0])
        conversation = [
            {
                "role": "user",
                "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
            },
            {"role": "assistant", "content": "OK"},
            # *icl,
            {"role": "user", "content": obs[0][len("-= Welcome to TextWorld, ALFRED! =-\n\n") :]},
        ]
        for _ in range(40):
            prompt: str = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            # result = a.generate(prompt, params)
            # action_text = result[0].outputs[0].text
            action_text = gen_two_line(a, prompt)
            print(action_text)
            conversation.append({"role": "assistant", "content": action_text})
            try:
                action = action_text.split("Action: ")[1]
            except:
                invalid += 1
                break

            obs, scores, dones, infos = env.step([action])
            ob = obs[0]
            if ob.startswith("You arrive at loc "):
                ob = ob[ob.find(". ") + 2 :]
            conversation.append({"role": "user", "content": "Observation: " + ob})
            print(obs[0])
            # if obs[0] == "Nothing happens.":
            #     invalid += 1
            #     break

            if dones[0]:
                success += scores[0]
                break
        pbar.set_description("Success: {} Rate: {:.2f} Invalid: {}".format(success, success / (i + 1), invalid))

print(success)
