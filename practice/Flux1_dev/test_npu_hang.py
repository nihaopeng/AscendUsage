#!/usr/bin/env python3
"""
复现 CANN 9.0.0-beta.1 / torch_npu 2.6.0 的逐个参数 .npu() 死锁 bug。

实测发现：
  - model.npu()             → ✅ 不会卡，内部 _apply 有某种批处理/顺序优化
  - 逐个 p.data = p.data.npu() → 🐛 会卡住！NPU 驱动内部死锁
  - flat buffer 单次搬运    → ✅ 正常，且最快

问题: 当模型参数通过 meta+assign 路径创建（如 load_flow_model），
     每个参数有独立 storage，逐个 .npu() 时 NPU 驱动会内部死锁。

用法:
    python test_npu_hang.py                 # 全部测试
    python test_npu_hang.py --skip-hang     # 跳过逐个搬运的死锁测试
    python test_npu_hang.py --timeout 30    # 自定义超时秒数

依赖: 仅 torch + torch_npu，无需 flux 或其他包。
"""

import argparse
import gc
import signal
import sys
import time

import torch
import torch.nn as nn
import torch_npu


# ═══════════════════════════════════════════════════════════════════════════════
# 信号超时 — 用于检测 NPU 搬运是否死锁
# ═══════════════════════════════════════════════════════════════════════════════

class Hangs(Exception):
    """NPU 搬运超时 → bug 已复现"""
    pass


def _on_timeout(signum, frame):
    raise Hangs("NPU 搬运在超时时间内未返回 → BUG 已复现!")


# ═══════════════════════════════════════════════════════════════════════════════
# 构建一个与 Flux 参数规模相当的合成模型
# ═══════════════════════════════════════════════════════════════════════════════

def build_synthetic_model(num_params: int = 780) -> nn.Module:
    """
    构建一个参数量级与 Flux (~12B 参数, ~780 个参数张量) 类似的合成模型。

    关键: 必须模拟 load_flow_model() 的真实路径:
      1. torch.device("meta") 上创建模型结构
      2. 模拟 safetensors 逐个加载: 每个参数是独立分配的 CPU tensor
      3. load_state_dict(assign=True) 物化参数
      4. del state_dict 释放中间引用

    这与直接 nn.Linear 创建（测试脚本旧方案）有本质区别:
      - 直接创建: PyTorch 的权重初始化在同一个内存池中分配，参数 storage 可能共享
      - meta+assign: 每个参数的 storage 来自独立分配的 CPU tensor，完全独立

    在 CANN 9.0.0-beta.1 上，独立 storage 的数百个参数逐个 .npu() 时会触发驱动死锁。
    """
    linear_specs = []
    # 大参数 (模拟 single_blocks.linear1: 3072×21504 ≈ 66M)
    for _ in range(38):
        linear_specs.append((21504, 3072))   # weight ~66M
        linear_specs.append((3072,))          # bias

    # 中等参数 (模拟 double_blocks 内的主要 linear: 3072×3072 ≈ 9.4M)
    for _ in range(19 * 6):
        linear_specs.append((3072, 3072))    # weight ~9.4M
        linear_specs.append((3072,))          # bias

    # 小参数 (norm, embedding, head, img_in, time_in, etc.)
    for _ in range(19 * 4):
        linear_specs.append((3072, 128))     # ~393K
        linear_specs.append((3072,))
    for _ in range(30):
        linear_specs.append((3072, 64))      # ~197K
        linear_specs.append((3072,))

    # 截断到 ~780 params
    linear_specs = linear_specs[:num_params]

    # Step 1: 在 meta device 上创建模型结构（与 Flux(model.py) 一致）
    with torch.device("meta"):
        model = nn.Sequential()
        for i, spec in enumerate(linear_specs):
            if len(spec) == 2:
                layer = nn.Linear(spec[0], spec[1], bias=True)
            else:
                layer = nn.Linear(spec[0], spec[0], bias=False)
            model.add_module(f"layer_{i}", layer)
    model = model.to(torch.bfloat16)  # Flux 使用 bf16

    # Step 2: 模拟 safetensors 逐个加载——每个参数是独立分配的 CPU tensor
    #         与 load_sft_with_tqdm() 中的 safe_open(device="cpu") + get_tensor() 对应
    state_dict = {}
    for name, p in model.named_parameters():
        state_dict[name] = torch.randn(p.shape, dtype=torch.bfloat16, device="cpu")

    # Step 3: load_state_dict(assign=True) 物化 meta 参数
    #         对应 util.py 第 734 行
    model.load_state_dict(state_dict, strict=False, assign=True)

    # Step 4: 删除 state_dict，释放中间引用（对应 util.py 第 735 行 del sd）
    del state_dict

    # 计算实际参数量
    total = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in model.parameters())
    print(f"  合成模型: {n_tensors} 个参数张量, "
          f"{total:,} 元素, ~{total * 2 / 1e9:.2f} GB (bf16)")
    print(f"  创建路径: meta→独立CPU分配→assign=True (模拟 safetensors 加载)")

    return model


def model_stats(model: nn.Module) -> dict:
    """统计模型参数分布"""
    total = 0
    n_tensors = 0
    largest = (None, 0)
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        n_tensors += 1
        if n > largest[1]:
            largest = (name, n)
    return {
        "total_params": total,
        "num_tensors": n_tensors,
        "size_gb": total * 2 / 1e9,  # bf16 = 2 bytes
        "largest_name": largest[0],
        "largest_elements": largest[1],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 三种搬运方式对比
# ═══════════════════════════════════════════════════════════════════════════════

def approach_a_model_npu(model: nn.Module, timeout: float) -> str:
    """
    [A] model.npu() — 带超时检测。内部 _apply 可能有优化，实测不一定卡。
    """
    print(f"  [A] model.npu() (超时={timeout}s)...", flush=True)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(timeout))

    t0 = time.time()
    try:
        model.npu()
        elapsed = time.time() - t0
        signal.alarm(0)
        on_npu = sum(1 for _, p in model.named_parameters() if p.device.type == "npu")
        print(f"  [A] ✅ {elapsed:.1f}s 完成, {on_npu} params 在 NPU")
        return "SUCCESS"
    except Hangs:
        signal.alarm(0)
        print(f"  [A] 🐛 超时! model.npu() 死锁 bug 已复现!")
        return "HUNG (bug confirmed)"
    except Exception as e:
        signal.alarm(0)
        print(f"  [A] ❌ 异常: {type(e).__name__}: {e}")
        return f"ERROR: {type(e).__name__}"


def approach_b_one_by_one(model: nn.Module, timeout: float) -> str:
    """
    [B] 逐个 p.data = p.data.npu() — 实测在 CANN 9.0.0-beta.1 上会死锁！
        每个参数有独立 storage 时，逐个搬运会触发 NPU 驱动内部死锁。
    """
    print(f"  [B] 逐个 p.data = p.data.npu() (超时={timeout}s, 预期会卡住)...", flush=True)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(timeout))

    params = list(model.named_parameters())
    count = 0
    t0 = time.time()
    try:
        for name, p in params:
            p.data = p.data.npu()
            count += 1
            if count % 50 == 0:
                print(f"      {count}/{len(params)} 完成 ({time.time() - t0:.1f}s)", flush=True)
        elapsed = time.time() - t0
        signal.alarm(0)
        on_npu = sum(1 for _, p in model.named_parameters() if p.device.type == "npu")
        print(f"  [B] ✅ {elapsed:.1f}s 完成, {on_npu} params 在 NPU (未复现)")
        return "SUCCESS"
    except Hangs:
        signal.alarm(0)
        print(f"  [B] 🐛 卡在第 {count}/{len(params)} 个参数! 逐个搬运死锁 bug 已复现!")
        return f"HUNG at param {count}/{len(params)} (bug confirmed)"
    except Exception as e:
        signal.alarm(0)
        print(f"  [B] ❌ 异常在第 {count} 个参数: {type(e).__name__}: {e}")
        return f"ERROR: {type(e).__name__}"


def approach_c_flat_buffer(model: nn.Module) -> str:
    """
    [C] Flat buffer — 所有参数拼成连续 buffer，1 次 .npu() 搬运，视图切片。
    """
    print("  [C] Flat buffer 单次搬运...", flush=True)

    t0 = time.time()

    # 1. 统计布局
    param_info = []
    total_elems = 0
    for name, p in model.named_parameters():
        n = p.numel()
        param_info.append((name, n, p.shape, p.dtype))
        total_elems += n

    # 2. 拼接所有 CPU 参数到一个连续 buffer
    flat_cpu = torch.empty(total_elems, dtype=torch.bfloat16, device="cpu")
    offset = 0
    for p in model.parameters():
        n = p.numel()
        flat_cpu[offset:offset + n].copy_(p.data.view(-1))
        offset += n
    t_copy = time.time() - t0
    print(f"      CPU 拼接: {t_copy:.2f}s", flush=True)

    # 3. 一次性搬运到 NPU（核心：只调 1 次 .npu()）
    t1 = time.time()
    flat_npu = flat_cpu.npu()
    t_transfer = time.time() - t1
    del flat_cpu
    print(f"      NPU 搬运: {t_transfer:.2f}s (单次)", flush=True)

    # 4. 视图切片替换各参数
    t2 = time.time()
    offset = 0
    for (name, n, shape, dtype), p in zip(param_info, model.parameters()):
        p.data = flat_npu[offset:offset + n].view(shape).to(dtype)
        offset += n
    t_slice = time.time() - t2
    print(f"      视图切片: {t_slice:.2f}s", flush=True)

    # 保持 buffer 存活
    model._flat_npu_buffer = flat_npu
    torch.npu.synchronize()

    total_t = time.time() - t0

    # 验证
    on_npu = sum(1 for _, p in model.named_parameters() if p.device.type == "npu")
    total_p = sum(1 for _ in model.parameters())
    print(f"  [C] ✅ 总耗时 {total_t:.1f}s, {on_npu}/{total_p} 参数在 NPU")
    return "SUCCESS"


def approach_c_flat_buffer(model: nn.Module) -> str:
    """
    [C] Flat buffer — 所有参数拼成连续 buffer，1 次 .npu() 搬运，视图切片。
    """
    print("  [C] Flat buffer 单次搬运...", flush=True)

    t0 = time.time()

    # 1. 统计布局
    param_info = []
    total_elems = 0
    for name, p in model.named_parameters():
        n = p.numel()
        param_info.append((name, n, p.shape, p.dtype))
        total_elems += n

    # 2. 拼接所有 CPU 参数到一个连续 buffer
    flat_cpu = torch.empty(total_elems, dtype=torch.bfloat16, device="cpu")
    offset = 0
    for p in model.parameters():
        n = p.numel()
        flat_cpu[offset:offset + n].copy_(p.data.view(-1))
        offset += n
    t_copy = time.time() - t0
    print(f"      CPU 拼接: {t_copy:.2f}s", flush=True)

    # 3. 一次性搬运到 NPU（核心：只调 1 次 .npu()）
    t1 = time.time()
    flat_npu = flat_cpu.npu()
    t_transfer = time.time() - t1
    del flat_cpu
    print(f"      NPU 搬运: {t_transfer:.2f}s (单次)", flush=True)

    # 4. 视图切片替换各参数
    t2 = time.time()
    offset = 0
    for (name, n, shape, dtype), p in zip(param_info, model.parameters()):
        p.data = flat_npu[offset:offset + n].view(shape).to(dtype)
        offset += n
    t_slice = time.time() - t2
    print(f"      视图切片: {t_slice:.2f}s", flush=True)

    # 保持 buffer 存活
    model._flat_npu_buffer = flat_npu
    torch.npu.synchronize()

    total_t = time.time() - t0

    # 验证
    on_npu = sum(1 for _, p in model.named_parameters() if p.device.type == "npu")
    total_p = sum(1 for _ in model.parameters())
    print(f"  [C] ✅ 总耗时 {total_t:.1f}s, {on_npu}/{total_p} 参数在 NPU")
    return "SUCCESS"


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="复现 CANN 逐个参数 .npu() 死锁 bug (纯 torch + torch_npu)"
    )
    parser.add_argument("--skip-hang", action="store_true",
                        help="跳过 A/B 死锁复现测试")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="超时秒数 (default: 60)")
    args = parser.parse_args()

    print("=" * 60)
    print("CANN 逐个参数 .npu() 死锁 Bug 复现")
    print("=" * 60)

    # 环境信息
    print(f"\nPyTorch: {torch.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")
    if torch.npu.is_available():
        print(f"NPU count: {torch.npu.device_count()}")
        print(f"NPU device: {torch.npu.get_device_name(0)}")

    # ── 构建合成模型 ──────────────────────────────────────────────
    print("\n构建合成模型...", flush=True)
    model = build_synthetic_model(num_params=780)
    stats = model_stats(model)
    print(f"最大参数: {stats['largest_name']} "
          f"({stats['largest_elements']:,} elements, "
          f"{stats['largest_elements'] * 2 / 1e6:.1f} MB)")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()

    result_a = "SKIPPED"
    result_b = "SKIPPED"

    # ── 测试 A: model.npu() ────────────────────────────────────────
    if args.skip_hang:
        print("\n(跳过死锁测试 A 和 B)")
    else:
        print(f"\n{'─' * 60}")
        print(f"测试 [A]: model.npu() — 若卡住 {args.timeout}s 则确认 bug")
        print(f"{'─' * 60}")
        result_a = approach_a_model_npu(model, timeout=args.timeout)

        # 无论 A 是否卡住，都重建干净模型给 B
        del model
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        model = build_synthetic_model(num_params=780)
        gc.collect()

        # ── 测试 B: 逐个搬运 ──────────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"测试 [B]: 逐个 p.data = p.data.npu() — 若卡住 {args.timeout}s 则确认 bug")
        print(f"{'─' * 60}")
        result_b = approach_b_one_by_one(model, timeout=args.timeout)

    # ── 测试 C: Flat buffer（总是从干净模型开始）──────────────────
    del model
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    model = build_synthetic_model(num_params=780)
    gc.collect()

    print(f"\n{'─' * 60}")
    print(f"测试 [C]: Flat buffer 单次搬运 (解决方案)")
    print(f"{'─' * 60}")
    result_c = approach_c_flat_buffer(model)

    # ── 总结 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("总结")
    print(f"{'=' * 60}")
    print("  实测结论:")
    status_a = "卡住" if "HUNG" in str(result_a) else ("跳过" if result_a == "SKIPPED" else "正常")
    status_b = "卡住" if "HUNG" in str(result_b) else ("跳过" if result_b == "SKIPPED" else "正常")
    print(f"    [A] model.npu()              → {status_a}")
    print(f"    [B] 逐个 p.data = p.data.npu() → {status_b}")
    print(f"    [C] Flat buffer 单次搬运     → 正常 (解决方案)")
    print()
    print("  Bug 触发条件:")
    print("    1. 模型通过 meta device + load_state_dict(assign=True) 物化")
    print("    2. 每个参数有独立 storage（模拟 safetensors 逐个加载）")
    print("    3. 逐个 p.data = p.data.npu() 搬运 → NPU 驱动内部死锁")
    print("  解决方案: 将所有参数拼成 1 个连续 buffer，")
    print("           仅调 1 次 .npu()，然后视图切片还原。")


if __name__ == "__main__":
    main()
