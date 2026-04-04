// csrc/kernels/add_custom.cpp
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "add.h"

namespace ascendc_ops {
    extern "C" {
        at::Tensor ascend_scalar_vadd(const at::Tensor& self, const at::Tensor& other) {
            // 获取 PyTorch 当前使用的硬件 Stream
            // 注意：PrivateUse1 对应的 Stream 获取方式
            auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
            // 分配Device侧输出内存
            auto result = at::empty_like(self);
            // 提取设备原始指针 (Device Pointer)
            // data_ptr<T>() 会返回 Tensor 在硬件上的地址
            uint8_t* a_ptr = static_cast<uint8_t*>(self.mutable_data_ptr());
            uint8_t* b_ptr = static_cast<uint8_t*>(other.mutable_data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(result.mutable_data_ptr());
            int len = (int)self.numel();
            // 调用 Launch 函数
            // 你需要根据硬件设计传递 blockDim, stream 和 tiling 参数
            uint32_t blockDim = 8; // 比如开启 8 个核并行
            // 注意：这里需要根据你真实的 launch_vector_add 签名来改
            // 这里的 launch 内部会执行 <<<blockDim, ctrl, acl_stream>>>
            launch_scalar_vadd(aclStream, blockDim, a_ptr, b_ptr, out_ptr, len);
            return result;
        }

        at::Tensor ascend_vector_vadd(const at::Tensor& self, const at::Tensor& other) {
            auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
            auto result = at::empty_like(self);
            uint8_t* a_ptr = static_cast<uint8_t*>(self.mutable_data_ptr());
            uint8_t* b_ptr = static_cast<uint8_t*>(other.mutable_data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(result.mutable_data_ptr());
            int len = (int)self.numel();
            uint32_t blockDim = 8; // 根据实际情况调整
            launch_vector_vadd(aclStream, blockDim, a_ptr, b_ptr, out_ptr, len);
            return result;
        }
    }
}// namespace ascendc_ops