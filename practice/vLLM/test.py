from vllm import LLM, SamplingParams



prompts = [

    "Hello, my name is",

    "The president of the United States is",

    "The capital of France is",

    "The future of AI is",

]



# Create a sampling params object.

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.

llm = LLM(
    model="/mnt/nvme1p1/pengyt/Qwen/Qwen3-0.6B",
    # Avoid mmap-backed tensors on this host. The Ascend driver fails to pin
    # those pages during the host-to-NPU weight copy and leaves EngineCore
    # stuck in the kernel.
    safetensors_load_strategy="eager",
)



# Generate texts from the prompts.

outputs = llm.generate(prompts, sampling_params)

for output in outputs:

    prompt = output.prompt

    generated_text = output.outputs[0].text

    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
