import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/ckpt/minicpm_alfworld_sft", trust_remote_code=True)
a = AutoModelForCausalLM.from_pretrained("/mnt/data/ckpt/minicpm_alfworld_sft", trust_remote_code=True)
a.to(torch.bfloat16)
a.to("cuda")

conversation = [
    {
        "role": "user",
        "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
    },
    {"role": "assistant", "content": "OK"},
    {
        "role": "user",
        "content": "You are in the middle of a room. Looking quickly around you, you see a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\n\nYour task is to: clean some potato and put it in countertop.",
    },
    {
        "role": "assistant",
        "content": "Thought: I need to find a potato to clean and place it on the countertop. I should start by looking for a potato in the room.\nAction: go to countertop 1",
    },
    {"role": "user", "content": "Observation: On countertop 1, you see a tomato 1."},
    {
        "role": "assistant",
        "content": "Thought: There's a tomato on the countertop. I need to remove it before cleaning the potato.\nAction: go to drawer 2",
    },
    {"role": "user", "content": "Observation: In drawer 2, you see a potato 1."},
    {
        "role": "assistant",
        "content": "Thought: I found a potato in drawer 2. I should clean it before placing it on the countertop.\nAction: clean potato 1 with drawer 2",
    },
    {"role": "user", "content": "Observation: Nothing happened."},
    {
        "role": "assistant",
        "content": "Thought: Cleaning the potato is complete. Now I should go to countertop 1 to place the potato.\nAction: go to countertop 1",
    },
    {"role": "user", "content": "Observation: On countertop 1, you see a tomato 1."},
    {
        "role": "assistant",
        "content": "Thought: I am now at countertop 1. I should place the potato on the countertop.\nAction: put potato 1 in/on countertop 1",
    },
    {"role": "user", "content": "Observation: Nothing happened."},
]

prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
input_tokens = tokenizer(prompt, return_tensors="pt")
input_tokens.to("cuda")
result = a.generate(input_tokens["input_ids"], max_new_tokens=2048)

print(tokenizer.decode(result[0]))
