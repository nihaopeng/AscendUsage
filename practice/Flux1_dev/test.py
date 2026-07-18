import torch
import torch_npu

x = torch.randn(1024, 1024, dtype=torch.bfloat16)
print(x.dtype)

y = x.to("npu")

print(y.device)