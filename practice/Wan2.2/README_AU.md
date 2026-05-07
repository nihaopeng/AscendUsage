# practice of Wan2.2

## install

`conda create -n diffusers python=3.11`

`conda activate diffusers`

`pip install -r requirements.txt`

`pip install "huggingface_hub[cli]"`

`wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip3 install torch-2.6.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`

`wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post5-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip3 install torch_npu-2.6.0.post5-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip install diffusers torchvision==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

+ decord

`conda install -c conda-forge ffmpeg==4.4 -y`

`git clone --recursive https://github.com/dmlc/decord`

`cd decord`

`mkdir build && cd build`

`cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX`

`make`

`cd ../python`

`python setup.py install --no-cache-dir`

## run

### download model parameters

`hf download Wan-AI/Wan2.2-I2V-A14B --local-dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B`

### run image-to-video

+ add below code into generate.py at the first line, this is a tool that can translate device into npu automatically.

```python
+ from torch_npu.contrib import transfer_to_npu
```

+ large discontinuous memory alloc for fsdp

wan/distributed/fsdp.py [24]
```python
+ model = model.to(torch.bfloat16)
```
actually, FSDP will automatically translate it into bfloat16, but, here occurs some problem for ascend-torch.

+ datatype adaptation

wan/modules/model.py [239,471]
wan/distributed/sequence_parallel.py [126]
```python
+ if e.dtype != torch.float32:
+         e = e.to(torch.float32)
+ if e0.dtype != torch.float32:
+         e0 = e0.to(torch.float32)
```

wan/distributed/sequence_parallel.py [57]
wan/modules/model.py [54]
```python
+ freqs = [t.to(torch.complex64) if t.dtype == torch.complex128 else t for t in freqs]
+ f, h, w = int(f), int(h), int(w)
```

wan/distributed/sequence_parallel.py [9]
```python
+ def pad_freqs(original_tensor, target_len):
+     seq_len, s1, s2 = original_tensor.shape
+     pad_size = target_len - seq_len
+     if pad_size <= 0:
+         return original_tensor
+     padding_tensor = torch.ones(
+         pad_size,
+         s1,
+         s2,
+         dtype=torch.float32, 
+         device=original_tensor.device
+     )
+     padding_tensor = padding_tensor.to(original_tensor.dtype)
+     padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
+     return padded_tensor

- def pad_freqs(original_tensor, target_len):
-     seq_len, s1, s2 = original_tensor.shape
-     pad_size = target_len - seq_len
-     padding_tensor = torch.ones(
-         pad_size,
-         s1,
-         s2,
-         dtype=original_tensor.dtype,
-         device=original_tensor.device)
-     padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
-     return padded_tensor
```

+ flash_attn

wan/modules/attention.py [54]
```python
+ assert q.device.type in ['cuda', 'npu'] and q.size(-1) <= 256
+ if q.device.type == 'npu':
+         import torch.nn.functional as F
+         orig_shape = q.shape
+         out_dtype = q.dtype
+         if q.dim() == 3:
+             q = q.unsqueeze(0)
+             k = k.unsqueeze(0)
+             v = v.unsqueeze(0)
+         q_npu = q.transpose(1, 2)
+         k_npu = k.transpose(1, 2)
+         v_npu = v.transpose(1, 2)
+         scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
+         out = F.scaled_dot_product_attention(
+             q_npu, k_npu, v_npu, 
+             attn_mask=None,
+             dropout_p=dropout_p,
+             is_causal=causal,
+             scale=scale
+         )
+         out = out.transpose(1, 2)
+         if len(orig_shape) == 3:
+             out = out.squeeze(0)
+         return out.to(out_dtype)
```

+ run generation

`python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."`