// csrc/kernels/add_custom.cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "add.h"

namespace ascendc_ops {
    extern "C" {
        at::Tensor ascend_matmul(const at::Tensor& self, const at::Tensor& other) {
            auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
            auto a = self.contiguous();// 确保输入内存连续
            auto b = other.contiguous();
            auto result = at::empty_like(self);
            uint8_t* a_ptr = static_cast<uint8_t*>(self.mutable_data_ptr());
            uint8_t* b_ptr = static_cast<uint8_t*>(other.mutable_data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(result.mutable_data_ptr());
            uint32_t m = self.size(0);      // 第一维：行
            uint32_t k = self.size(1);      // 第二维：共有维度
            uint32_t n = other.size(1);     // 第二矩阵的第二维：列
            uint32_t blockDim = 8;
            launch_matmul(aclStream, blockDim, a_ptr, b_ptr, out_ptr, m, k, n);
            return result;
        }
    }
}// namespace ascendc_ops