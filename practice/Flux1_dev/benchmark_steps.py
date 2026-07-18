"""测试不同 num_inference_steps（序列步数）下，无 compile 与有 compile 的性能差异。

num_inference_steps 决定了去噪循环的迭代次数（即 transformer 被调用的次数），
步数越多，编译带来的单步加速被放大的效果越明显。
"""

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import time
from diffusers import FluxPipeline
import torchair
from torchair.configs.compiler_config import CompilerConfig
import torch.utils._triton
torch.utils._triton.has_triton = lambda: False

# ===== 配置 =====
config = CompilerConfig()
config.experimental_config.frozen_parameter = True
config.experimental_config.tiling_schedule_optimize = True
config.mode = "max-autotune"
npu_backend = torchair.get_npu_backend(compiler_config=config)

# ===== 加载模型 =====
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
print("Model loaded on CPU, moving to NPU...")
pipe.to("npu")
original_transformer = pipe.transformer

prompt = "a tiny astronaut hatching from an egg on the moon"
height, width = 768, 1360  # 固定分辨率，排除形状变化干扰

step_counts = [10, 20, 30, 40, 50]

no_compile_times = []    # [steps] → 无 compile 推理时间
compile_times = []        # [steps] → 编译耗时（仅首次）
compiled_times = []       # [steps] → 编译后推理时间（稳定）

# ========== 1. 无 compile：每个 step 数跑一次（warm-up 后计时） ==========
print("\n" + "=" * 60)
print("Phase 1: Without torch.compile")
print("=" * 60)
for steps in step_counts:
    common_kwargs = dict(prompt=prompt, guidance_scale=3.5, height=height,
                         width=width, num_inference_steps=steps)

    # warm-up（不计时，预热 CANN JIT）
    _ = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()

    t0 = time.time()
    _ = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()
    elapsed = time.time() - t0
    no_compile_times.append(elapsed)
    print(f"  steps={steps:3d}: {elapsed:.2f}s  ({elapsed/steps:.2f}s/step)")

# ========== 2. 恢复干净模块，compile 一次 ==========
print("\n" + "=" * 60)
print("Phase 2: torch.compile + inference")
print("=" * 60)

pipe.transformer = original_transformer
torch._dynamo.reset()

torch.npu.synchronize()
t0 = time.time()
pipe.transformer = torch.compile(pipe.transformer, backend=npu_backend)
torch.npu.synchronize()
compile_only = time.time() - t0
print(f"  Compile time: {compile_only:.2f}s")

# warm-up（编译后的首次执行有图下沉开销，不计时）
warm_kwargs = dict(prompt=prompt, guidance_scale=3.5, height=height,
                   width=width, num_inference_steps=20)
_ = pipe(**warm_kwargs).images[0]     # warm-up
_ = pipe(**warm_kwargs).images[0]     # 第二次才是稳定态
torch.npu.synchronize()

# ========== 3. 编译后：每个 step 数跑一次 ==========
for steps in step_counts:
    common_kwargs = dict(prompt=prompt, guidance_scale=3.5, height=height,
                         width=width, num_inference_steps=steps)

    t0 = time.time()
    _ = pipe(**common_kwargs).images[0]
    torch.npu.synchronize()
    elapsed = time.time() - t0
    compiled_times.append(elapsed)
    print(f"  steps={steps:3d}: {elapsed:.2f}s  ({elapsed/steps:.2f}s/step)")

# ========== 汇总 ==========
print(f"\n{'='*70}")
print(f"Summary: 固定分辨率 {height}x{width}，不同推理步数下的性能对比")
print(f"  Compile time (one-time): {compile_only:.2f}s")
print(f"{'='*70}")
print(f"{'Steps':<8} {'No Compile':<14} {'Compiled':<14} {'Speedup':<12} {'NoC/step':<12} {'C/step':<12}")
print(f"{'-'*70}")
for i, steps in enumerate(step_counts):
    speedup = no_compile_times[i] / compiled_times[i]
    print(f"  {steps:<8} {no_compile_times[i]:<14.2f} {compiled_times[i]:<14.2f} "
          f"{speedup:<12.2f} {no_compile_times[i]/steps:<12.2f} {compiled_times[i]/steps:<12.2f}")

avg_speedup = sum(n / c for n, c in zip(no_compile_times, compiled_times)) / len(step_counts)
print(f"{'='*70}")
print(f"  Average speedup: {avg_speedup:.2f}x")
print(f"  Amortized compile cost break-even: ~{compile_only / (no_compile_times[0]/step_counts[0] - compiled_times[0]/step_counts[0]):.0f} steps")
print(f"{'='*70}")
