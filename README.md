# Offline Reinforcement Learning for LLM Multi-Step Reasoning

Source code for [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)

Model: [Policy](https://huggingface.co/jwhj/Qwen2.5-Math-1.5B-OREO) | [Value](https://huggingface.co/jwhj/Qwen2.5-Math-1.5B-OREO-Value)

![Image goes here](./OREO.png)

# Installation

This repo is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and the installation follows a similar process. We recommend using Docker to setup the environment.

First build Docker image
```bash
cd dockerfile
docker build -t [IMAGE_NAME] .
```

Start a docker container
```bash
docker run -itd --ipc host --gpus all [IMAGE_NAME] bash
```

Attach to the container
```bash
docker exec -it [CONTAINER_ID] /bin/bash
```

Install the current repo
```bash
cd [PATH_TO_THIS_REPO]
pip install -e .
```

As the data collection process involves randomness, we will publish the training data used in our experiments in the near future.

# Reproduction
## Training
You may need to change the following command line options in the following scripts:
- `--train_file` specifies the path of training data in OREO experiments.
- `--dataset` specifies the path of training data in SFT experiments.
- `--save_path` specifies the path to save the model.
- `--pretrain` specifies the path to load the pretrained model. In OREO experiments, this should be the path to the SFT model.

### Math Reasoning

Supervised fine-tuning
```bash
cd example/scripts
bash train_oreo_sft.sh
```

OREO training
```bash
cd example/scripts
bash train_oreo.sh
```

To train the `DeepSeekMath-7B-Instruct` model,
```bash
cd example/scripts
bash train_oreo_deepseek-math.sh
```
Note that `DeepSeekMath-7B-Instruct` is already supervise fine-tuned, so we don't have an SFT phase here.

### ALFWorld

Supervised fine-tuning
```bash
cd example/scripts
bash train_oreo_alfworld_sft.sh
```

OREO training
```bash
cd example/scripts
bash train_oreo_alfworld.sh
```

## Evaluation
### Math Reasoning

Make sure you have `antlr4-python3-runtime==4.11.0` installed.

For Qwen-based models
```bash
cd example/scripts
python ../scratch/run_qwen.py --model [PATH_TO_YOUR_MODEL] --save [SAVE_GENERATED_RESULTS_JSONL]
```

For DeepSeekMath-based models
```bash
cd example/scripts
python ../scratch/run_qwen.py --model [PATH_TO_YOUR_MODEL] --no_bos --save [SAVE_GENERATED_RESULTS_JSONL]
```
Note the `--no_bos` option here.

### ALFWorld

This part requires [ALFWorld](https://github.com/alfworld/alfworld) to be installed.

First start an vllm server
```bash
python -m vllm.entrypoints.openai.api_server --model [PATH_TO_YOUR_MODEL]
```

Then run evaluation with
```bash
cd example/scripts
python ../scratch/run_alfworld_async.py --model [PATH_TO_YOUR_MODEL] --save_dir [SAVE_GENERATED_TRAJS]
```
You can use `--split eval_in_distribution` for seen environments.
