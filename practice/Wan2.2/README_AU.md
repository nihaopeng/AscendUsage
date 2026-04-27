# practice of Wan2.2

## install

`conda create -n AscendUsage python=3.11`

`conda activate AscendUsage`

`pip install PyYAML`

`pip install "numpy<2"`

`pip install "setuptools<70"`

`pip install scipy decorator attrs psutil synr einops librosa peft`

+ torch
`wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip3 install torch-2.6.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`

+ torch_npu

`wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post5-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip3 install torch_npu-2.6.0.post5-cp311-cp311-manylinux_2_28_aarch64.whl`

`pip install -r requirements.txt`

`pip install "huggingface_hub[cli]"`

+ decord

`conda install ffmpeg=4.4 -c conda-forge`

`git clone --recursive https://github.com/dmlc/decord`

`cd decord`

`mkdir build && cd build`

`cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX`

`make`

`cd ../python`

`python3 setup.py install`

## run

### download model parameters

`hf download Wan-AI/Wan2.2-I2V-A14B --local-dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B`

### run image-to-video

+ add below code into generate.py at the first line, this is a tool that can translate device into npu automatically.

`from torch_npu.contrib import transfer_to_npu`

+ modified the diffusers

replace the code in $ENV/site-packages/diffusers/utils/torch_utils.py
```python
BACKEND_EMPTY_CACHE = {
        "cuda": torch.cuda.empty_cache,
        "xpu": torch.xpu.empty_cache,
        "cpu": None,
        "mps": torch.mps.empty_cache,
        "default": None,
}
```
into
```python
BACKEND_EMPTY_CACHE = {
        "cuda": torch.cuda.empty_cache,
        "npu": torch_npu.npu.empty_cache,
        "cpu": None,
        "mps": torch.mps.empty_cache,
        "default": None,
}
```
same with other code in $ENV/site-packages/diffusers/utils/torch_utils.py

+ insert code to avoid type checking

wan/modules/model.py[469]
```python
e = e.to(torch.float32)
e0 = e0.to(torch.float32)
```

+ 4,downgrade the type

wan/modules/model.py[51]
```python
if freqs.dtype == torch.complex128:
        freqs = freqs.to(torch.complex64)
x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64)... -> x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32)...
...reshape(seq_len, 1, -1) -> ...reshape(seq_len, 1, -1).to(torch.complex64)
```

wan/modules/attention.py[54]
```python
assert q.device.type in ['cuda', 'npu'] and q.size(-1) <= 256
    # 2. 针对 NPU 的分支处理
    if q.device.type == 'npu':
        import torch.nn.functional as F
        # Wan2.2 输入维度: [B, L, N, C]
        # F.scaled_dot_product_attention 期望维度: [B, N, L, C]
        # 调整维度顺序
        q_npu = q.transpose(1, 2)
        k_npu = k.transpose(1, 2)
        v_npu = v.transpose(1, 2)
        # 如果提供了 softmax_scale，手动应用它
        # 因为 F.scaled_dot_product_attention 默认使用 1/sqrt(d)
        # 如果项目传入了特定的 scale，这里需要处理一下
        scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
        # 执行计算
        out = F.scaled_dot_product_attention(
            q_npu, k_npu, v_npu, 
            attn_mask=None,
            dropout_p=dropout_p, 
            is_causal=causal,
            scale=scale
        )
        # 将结果从 [B, N, L, C] 转回 [B, L, N, C]
        return out.transpose(1, 2).to(q.dtype)
```
+ replace all 'nccl' to 'hccl'

+ run generation

`python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."`