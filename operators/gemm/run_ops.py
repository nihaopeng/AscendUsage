import torch
import torch_npu
import os
import time

# 1. 加载库 (确保 libdevice.so 已经包含了 ascend_matmul 的实现)
lib_path = os.path.abspath("libdevice.so")
torch.ops.load_library(lib_path)

def benchmark(op_func, name, iterations=100):
    # 预热
    for _ in range(10):
        op_func(a_npu, b_npu)
    torch_npu.npu.synchronize()
    
    start_time = time.perf_counter()
    res = None
    for _ in range(iterations):
        res = op_func(a_npu, b_npu)
    torch_npu.npu.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations * 1000 
    print(f"[{name}] Avg Time: {avg_time:.4f} ms")
    return res.cpu(), avg_time

def check_result(npu_out, cpu_out, name):
    max_diff = torch.max(torch.abs(npu_out - cpu_out)).item() 
    is_close = torch.allclose(npu_out, cpu_out, rtol=RTOL, atol=ATOL)
    print(f"[{name}] Max Diff: {max_diff:.8f}")
    if is_close:
        print(f"✅ [{name}] Passed!")
    else:
        print(f"❌ [{name}] Failed! (误差过大，请检查 Tiling 或精度)")
    return is_close

# 矩阵乘法建议使用稍微宽松一点的阈值，尤其是 FP32 累加时
RTOL = 1e-03
ATOL = 1e-03

# 2. 准备矩阵数据 (M, K) * (K, N)
M, K, N = 512, 512, 512  # 建议从 16 的倍数开始测试，便于硬件对齐
print(f"Creating matrices: ({M}x{K}) * ({K}x{N})...")

a_cpu = torch.randn(M, K, dtype=torch.float32)
b_cpu = torch.randn(K, N, dtype=torch.float32)

a_npu = a_cpu.npu()
b_npu = b_cpu.npu()

# 3. 性能测试与比对
print("-" * 30)
try:
    # 运行 NPU 自定义算子
    # 注意：根据你的 C++ 定义，调用路径为 torch.ops.ascendc_ops.ascend_matmul
    out_npu, time_npu = benchmark(
        torch.ops.ascendc_ops.ascend_matmul, "NPU Matmul", iterations=100
    )

    # 4. CPU 参考值 (标准矩阵乘)
    start_cpu = time.perf_counter()
    out_cpu = torch.matmul(a_cpu, b_cpu)
    end_cpu = time.perf_counter()
    time_cpu = (end_cpu - start_cpu) * 1000
    print(f"[CPU Matmul] Time: {time_cpu:.4f} ms")

    # 5. 结果校验
    check_result(out_npu, out_cpu, "NPU vs CPU")

    # 打印局部结果
    print(f"\nTop-left 3x3 Corner:")
    print(f"CPU:\n{out_cpu[:3, :3]}")
    print(f"NPU:\n{out_npu[:3, :3]}")

    print("-" * 30)
    print(f"🚀 Acceleration vs CPU: {time_cpu / time_npu:.2f}x")

except Exception as e:
    print("Error during benchmarking:", e)