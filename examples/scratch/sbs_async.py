import asyncio

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

template = "\nPlease reason step by step, and put your final answer within \\boxed{}."
chat_template = "User: {}\n\nAssistant:"

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        "/mnt/data/ckpt/dsm-inst_b2_lr1e-4_actor-lr5e-5_actor-loss-single-step_2epochs_merged",
        gpu_memory_utilization=0.5,
    )
)


async def main():
    result = engine.generate(
        {"prompt": chat_template.format("What is 1+1?" + template), "stream": False},
        sampling_params=SamplingParams(max_tokens=2048),
        request_id=0,
    )
    async for x in result:
        print(x)


engine.start_background_loop()

asyncio.run(main())
