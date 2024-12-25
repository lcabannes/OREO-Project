from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression

# correct derivation
# prompt = "User: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = " Wendi gives her chickens 15 cups of feed in the morning and 25 cups of feed in the afternoon, so in total, she gives them 15 + 25 = 40 cups of feed.\nEach chicken eats 3 cups of feed per day, so 20 chickens will eat 20 * 3 = 60 cups of feed per day.\nSince Wendi has already given her chickens 40 cups of feed, she needs to give them an additional 60 - 40 = 20 cups of feed in the final meal of the day.\nTherefore, Wendi needs to give her chickens 20 cups of feed in the final meal of the day.\nThe answer is $\\boxed{30}$."

# wrong derivation
prompt = "User: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
response = " Betty has $100/2 = $50 saved.\nHer grandparents gave her $15 x 2 = $30.\nBetty now has $50 + 30 = $80 saved.\nBetty needs $100 - $80 = $20 more to buy that wallet.\nSo the answer is $\\boxed{20}$."
# [0.408203125, 0.486328125, 0.50390625, -0.4375, -0.67578125]
# prompt = "User: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = " If Mark has 10 yellow flowers in his garden, and he has 80% more purple flowers, he has 10 * .8 = 8 more purple flowers than yellow flowers.\nSo, in total, Mark has 10 + 8 = <<10+8=18>>18 purple flowers.\nSince he has 25% as many green flowers as there are yellow and purple flowers, he has 10 + 18 * .25 = <<10+18*.25=22>>22 green flowers.\nTherefore, Mark has a total of 10 + 18 + 22 = <<10+18+22=50>>50 flowers in his garden.\nThe answer is $\\boxed{50}$."
# [0.4921875, 0.546875, 0.447265625, -0.53515625, -0.64453125]
# prompt = "User: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = " Wendi gives her chickens 15 cups of feed in the morning and 25 cups of feed in the afternoon, so in total, she gives them 15 + 25 = 40 cups of feed.\nEach chicken eats 3 cups of feed per day, so 20 chickens will eat 20 * 3 = 50 cups of feed per day.\nSince Wendi has already given her chickens 40 cups of feed, she needs to give them an additional 50 - 40 = 10 cups of feed in the final meal of the day.\nTherefore, Wendi needs to give her chickens 10 cups of feed in the final meal of the day.\nThe answer is $\\boxed{10}$."
# [0.443359375, 0.54296875, -0.6796875, -0.75, -0.765625]
# prompt = "User: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = "Let's break down the problem:\n\nKylar wants to buy 16 glasses. The first glass costs $5. For the remaining 15 glasses, each glass costs 60% of the price of the first glass. \n\nSo, the cost of the first glass is $5. \n\nThe cost of the remaining 15 glasses is 15 * $5 * 0.60 = $30. \n\nSo, the total cost of the 16 glasses is $5 + $30 = $35.\n\nSo the answer is $\\boxed{35}$."
# ??

# tests from math-shephred
# prompt = "User: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = """ Step 1: The total ratio representing the pizza is 5+8 = <<5+8=13>>13.
# Step 2: The waiter ate 13 x 8 / 13 = <<13*8/13=6>>6 slices of the pizza.
# Step 3: Buzz ate 78 - 6 = <<78-6=72>>72 slices of the pizza.
# Step 4: The waiter ate 20 less than the number of slices that Buzz ate which is 72 - 20 = 52.
# Step 5: The waiter ate 52 slices of the pizza.
# The answer is: 52"""
# [0.451171875, 0.2578125, -0.54296875, -0.455078125, -0.380859375, -0.41015625]
# prompt = "User: Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = """ Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons.
# She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week.
# She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week.
# Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week.
# She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year.
# The answer is: 13520"""
# [0.478515625, -0.0164794921875, -0.3359375, -0.162109375, -0.2412109375, -0.2255859375]
# """Val cuts a single watermelon into 40 slices, and each slice has 20 black seeds and 20 white seeds. What is the total number of seeds in a watermelon? Step 1: There are 20 x 40 = <<20*40=8000>>8000 black seeds. - Step 2: There are 20 x 40 = <<20*40=8000>>8000 white seeds. - Step 3: So, the total number of seeds is 8000 + 8000 = <<8000+8000=16000>>16000. The answer is: 16000 -"""
# prompt = "User: Val cuts a single watermelon into 40 slices, and each slice has 20 black seeds and 20 white seeds. What is the total number of seeds in a watermelon?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
# response = """ Step 1: There are 20 x 40 = <<20*40=8000>>8000 black seeds.
# Step 2: There are 20 x 40 = <<20*40=8000>>8000 white seeds.
# Step 3: So, the total number of seeds is 8000 + 8000 = <<8000+8000=16000>>16000.
# The answer is: 16000"""
# [0.451171875, -0.5625, -0.62109375, -0.6484375]

lines = response.split("\n")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
critic = get_llm_for_sequence_regression(
    "deepseek-ai/deepseek-math-7b-instruct",
    "critic",
    normalize_reward=False,  # TODO: maybe experiment with this layer
    use_flash_attention_2=True,
    bf16=True,
    load_in_4bit=False,
    lora_rank=64,
    lora_alpha=64,
    lora_dropout=0,
    target_modules="all-linear",
    # ds_config=strategy.get_ds_train_config(is_actor=True),
    zero_init_value_head=True,
)
critic.to("cuda")

# critic.load_adapter("/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_2epochs_critic", "default")
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_critic-only_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_clip_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_ema_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_clip_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_clip_2epochs_iter2_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_kl-reg_2epochs_critic", "default"
# )
# critic.load_adapter(
#     "/mnt/data/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_kl-reg1e-4_2epochs_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-single-step_kl-reg1e-5_2epochs_iter2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg1e-5_2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr1e-5_actor-loss-single-step_actor-freeze-ratio0-1_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg-unbiased1e-1_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg-unbiased1e-1_actor-freeze-ratio0-2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg-unbiased1e-2_actor-freeze-ratio0-2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_kl-reg-unbiased1e-3_actor-freeze-ratio0-2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-single-step_step-level_kl-reg-unbiased1e-3_actor-freeze-ratio0-2_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_b2_dart-b_lr1e-4_actor-lr5e-5_actor-loss-dro_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_full_b2_lr1e-4_actor-lr5e-6_actor-loss-dro_importance-sampling_rew01_2epochs_critic",
#     "default",
# )
# critic.load_adapter(
#     "/mnt/lustre/ckpt/dsm-inst_full_lr1e-4_actor-lr5e-6_beta0-03_rew01_actor-loss-dro_kl-reg-unbiased1e-2_plot-weights_2epochsf_critic",
#     "default",
# )
critic.load_adapter(
    "/mnt/lustre/ckpt/dsm-inst_full_lr1e-4_actor-lr5e-6_beta0-03_rew01_actor-loss-dro_critic-only_critic",
    "default",
)

input_token = tokenizer(
    prompt + response + " " + tokenizer.eos_token,
    max_length=2048,
    padding=False,
    truncation=True,
    return_tensors="pt",
)  # TODO
input_token.to("cuda")
offset = len(prompt)

values = critic(**input_token, action_mask=input_token["input_ids"])

results = []
for i, line in enumerate(lines):
    idx = input_token.char_to_token(offset)
    if idx is None:
        break
    offset += len(line) + 1
    results.append(values[0, idx - 1].item())

__import__("pdb").set_trace()
