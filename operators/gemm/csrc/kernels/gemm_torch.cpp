// csrc/kernels/add_custom.cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "gemm.h"

namespace ascendc_ops {
    extern "C" {
        at::Tensor ascend_gemm(const at::Tensor& self, const at::Tensor& other) {
            auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

            auto a = self.contiguous();// 确保输入内存连续
            auto b = other.contiguous();

            uint32_t m = a.size(0);      // 矩阵 A 的行数
            uint32_t k = a.size(1);      // 矩阵 A 的列数 / 矩阵 B 的行数
            uint32_t n = b.size(1);      // 矩阵 B 的列数
            
            auto result = at::empty({m, n}, at::TensorOptions().dtype(at::kFloat).device(a.device())); // 输出矩阵 C 的大小为 (m, n)，数据类型为 float32

            uint8_t* a_ptr = static_cast<uint8_t*>(self.mutable_data_ptr());
            uint8_t* b_ptr = static_cast<uint8_t*>(other.mutable_data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(result.mutable_data_ptr());

            uint32_t blockDim = 20;
            launch_gemm(aclStream, blockDim, a_ptr, b_ptr, out_ptr, m, k, n);
            return result;
        }
    }
}// namespace ascendc_ops