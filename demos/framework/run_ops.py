import torch
import torch_npu
import os
import time
# 1. 加载库
lib_path = os.path.abspath("libdevice.so")
torch.ops.load_library(lib_path)

def benchmark(op_func, name, iterations=100):
    # 预热
    for _ in range(10):
        op_func(a_npu, b_npu)
    torch_npu.npu.synchronize() # 确保预热完成
    
    # 开始计时
    start_time = time.perf_counter()
    res = None
    for _ in range(iterations):
        res = op_func(a_npu, b_npu)
    torch_npu.npu.synchronize() # 确保所有异步算子执行完毕
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations * 1000 # 毫秒
    print(f"[{name}] Avg Time: {avg_time:.4f} ms")
    return res.cpu(), avg_time

def check_result(npu_out, cpu_out, name):
    # 计算最大偏差
    max_diff = torch.max(torch.abs(npu_out - cpu_out)).item() 
    # 核心校验函数
    is_close = torch.allclose(npu_out, cpu_out, rtol=RTOL, atol=ATOL)
    print(f"[{name}] Max Diff: {max_diff:.8f}")
    if is_close:
        print(f"✅ [{name}] Passed! (within tol: rtol={RTOL}, atol={ATOL})")
    else:
        print(f"❌ [{name}] Failed! (exceeds tolerance)")
    return is_close

RTOL = 1e-04  # 相对误差
ATOL = 1e-05  # 绝对误差
# 2. 准备大批量数据 (数据量太小看不出 Vector 的优势，建议 10^6 以上)
print("Creating tensors...")
n = 1024*100 
a_cpu = torch.randn(n, dtype=torch.float32)
b_cpu = torch.randn(n, dtype=torch.float32)
a_npu = a_cpu.npu()
b_npu = b_cpu.npu()

# 3. 性能测试与比对
print("-" * 30)
try:
    print(f"Benchmarking with {n} elements...")
    # 测试 Vector 版本
    out_vector, time_vector = benchmark(
        torch.ops.ascendc_ops.ascend_vector_vadd, "Vector Op", iterations=100
    )
    # 测试 Scalar 版本
    out_scalar, time_scalar = benchmark(
        torch.ops.ascendc_ops.ascend_scalar_vadd, "Scalar Op", iterations=100
    )
    # 4. CPU 参考值
    out_cpu = a_cpu + b_cpu
    # 5. 结果校验
    match_scalar = check_result(out_scalar, out_cpu, "Scalar Op")
    match_vector = check_result(out_vector, out_cpu, "Vector Op")
    print(f"\nSample Values:")
    print(f"CPU: {out_cpu[:5].tolist()}")
    print(f"SCALAR: {out_scalar[:5].tolist()}")
    print(f"VECTOR: {out_vector[:5].tolist()}")
    diff = torch.abs(out_vector - out_cpu)
    max_val, max_idx = torch.max(diff, 0)
    print(f"最大误差值: {max_val.item()}, 发生在索引: {max_idx.item()}")
    print("-" * 30)
    print(f"🚀 Speedup (Vector vs Scalar): {time_scalar / time_vector:.2f}x")

except Exception as e:
    print("Error during benchmarking:", e)