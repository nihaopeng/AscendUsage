// csrc/ascend_interface.cpp
#include <torch/extension.h>

// 1. 声明算子 Schema
TORCH_LIBRARY(ascendc_ops, m) {
    m.def("ascend_matmul(Tensor self, Tensor other) -> Tensor");
}
// 2. 实现具体的内核（通常从其他文件引入）
extern "C" {
    at::Tensor ascend_matmul(const at::Tensor& self, const at::Tensor& other);
}
// 3. 绑定到你的硬件设备 (PrivateUse1)
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m) {
    m.impl("ascend_matmul", &ascend_matmul);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ascend_matmul", &ascend_matmul, "Ascend Matrix Multiply (NPU)");
}