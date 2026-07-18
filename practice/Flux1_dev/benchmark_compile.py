import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import time
from diffusers import FluxPipeline
import torchair
from torchair.configs.compiler_config import CompilerConfig
import torch.utils._triton
torch.utils._triton.has_triton = lambda: False

config = CompilerConfig()
config.experimental_config.frozen_parameter = True
config.experimental_config.tiling_schedule_optimize = True
config.mode = "max-autotune"

npu_backend = torchair.get_npu_backend(compiler_config=config)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)

print(f"Model loaded on CPU, moving to NPU...")
pipe.to("npu")

prompt = "a tiny astronaut hatching from an egg on the moon"

# 5 种不同形状，每种形状触发独立的 JIT/编译，避免缓存预热
shapes = [
    (768, 1360),
    (1024, 1024),
    (1360, 768),
    (1152, 896),
    (896, 1152),
]

num_runs = len(shapes)
# 保存原始 transformer 引用（不被 torch.compile 包装的干净版本）
original_transformer = pipe.transformer

no_compile_times = []
compile_times = []           # 仅编译耗时
compiled_warmup_times = []   # 编译后首次推理（warm-up，含图初始化开销）
compiled_infer_times = []    # 编译后第二次推理（稳定耗时）

for i, (height, width) in enumerate(shapes):
    common_kwargs = dict(
        prompt=prompt,
        guidance_scale=3.5,
        height=height,
        width=width,
        num_inference_steps=50,
    )

    # ===== 确保 transformer 是干净的原始版本 =====
    pipe.transformer = original_transformer
    torch._dynamo.reset()

    # ===== 1. 不 compile，直接推理 =====
    torch.npu.synchronize()
    t0 = time.time()
    out = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()
    no_elapsed = time.time() - t0
    no_compile_times.append(no_elapsed)

    # ===== 恢复干净 transformer，准备 compile 测试 =====
    pipe.transformer = original_transformer
    torch._dynamo.reset()

    # ===== 2. compile（仅计时编译）=====
    torch.npu.synchronize()
    t0 = time.time()
    pipe.transformer = torch.compile(pipe.transformer, backend=npu_backend)
    torch.npu.synchronize()
    compile_elapsed = time.time() - t0
    compile_times.append(compile_elapsed)

    # ===== 3. 编译后首次推理（warm-up，含图下沉等初始化开销）=====
    torch.npu.synchronize()
    t0 = time.time()
    out = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()
    warmup_elapsed = time.time() - t0
    compiled_warmup_times.append(warmup_elapsed)

    # ===== 4. 编译后第二次推理（稳定耗时）=====
    torch.npu.synchronize()
    t0 = time.time()
    out = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()
    infer_elapsed = time.time() - t0
    compiled_infer_times.append(infer_elapsed)

    print(f"[Shape {i+1}/{num_runs}] {height}x{width}")
    print(f"  Without compile:         {no_elapsed:.2f}s")
    print(f"  Compile only:            {compile_elapsed:.2f}s")
    print(f"  Compiled warm-up:        {warmup_elapsed:.2f}s")
    print(f"  Compiled stable:         {infer_elapsed:.2f}s")
    print()

# ========== 汇总 ==========
avg_no_compile = sum(no_compile_times) / num_runs
avg_compile = sum(compile_times) / num_runs
avg_compiled_warmup = sum(compiled_warmup_times) / num_runs
avg_compiled_infer = sum(compiled_infer_times) / num_runs

print(f"{'='*70}")
print(f"Summary (5 different shapes, each run is cold):")
print(f"  Shapes tested: {shapes}")
print(f"  Without compile (avg):             {avg_no_compile:.2f}s")
print(f"  Compile only (avg):                {avg_compile:.2f}s")
print(f"  Compiled warm-up (avg):            {avg_compiled_warmup:.2f}s")
print(f"  Compiled stable (avg):             {avg_compiled_infer:.2f}s")
print(f"  Speedup (no_compile / stable):     {avg_no_compile / avg_compiled_infer:.2f}x")
print(f"{'='*70}")
print(f"\nDetailed per-shape comparison:")
print(f"{'Shape':<14} {'No Compile':<14} {'Compile':<14} {'Warm-up':<14} {'Stable':<14}")
for idx, (h, w) in enumerate(shapes):
    print(f"  {h}x{w:<9} {no_compile_times[idx]:<14.2f} {compile_times[idx]:<14.2f} {compiled_warmup_times[idx]:<14.2f} {compiled_infer_times[idx]:<14.2f}")
