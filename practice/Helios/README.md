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

### compile mode, enable GE

`git clone https://gitcode.com/ascend/torchair.git`

`cd torchair && git submodule update --init --recursive`

`bash ./configure`

跳过输入ASCEND_SDK_PATH，因为不需要使用CPU模拟、

```bash
mkdir build
cd build
cmake ..
make torchair -j8
```

`make install_torchair`

## run

### download model parameters

`hf download BestWishYSH/Helios-Base --local-dir /mnt/nvme1p1/pengyt/BestWishYSH/Helios-Base`

### run with script

`export TORCH_NPU_DEVICE_CAPABILITY=9.0`

`bash scripts/inference/helios-base_i2v.sh`

## problem

+ 中间采样过程存在非正定可能，Ascend算子并没有做相关处理，这里加一个轻微扰动，让矩阵正定

{miniconda3_path}/envs/DiTs/lib/python3.11/site-packages/diffusers/schedulers/scheduling_helios_dmd.py
```python
eps = 1e-6
cov = cov + eps * torch.eye(
    cov.shape[-1],
    device=cov.device,
    dtype=cov.dtype,
)
```

+ Ascend算子不支持float64，因此遇见类型错误，需要手动将库中某些数据转换。

+ coreDim超过65535，导致triton kernel launch失败，修改triton kernel launch参数。
```python
_apply_rope_transposed_kernel[(n_rows,)](
                x_flat,
                out,
                cos,
                sin,
                n_heads,
                x_flat.stride(0),
                out.stride(0),
                cos.stride(0),
                head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
```
n_rows=441600直接导致报错

给_apply_rope_transposed_kernel增加一个offset参数，表示当前批次的偏移量，分批执行。

修正之后for循环调用执行_apply_rope_transposed_kernel一次调用花费30ms，大概占整个forward流程的50%

第二种修复方案是让triton的kernel一次处理多个head：

```python
@triton.jit
def _apply_rope_transposed_kernel(
    X,
    Out,
    cos,
    sin,
    n_heads: tl.constexpr,
    n_rows,
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    stride_freq: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * BLOCK_ROWS

    half_head_dim = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < half_head_dim

    # BLOCK_ROWS 是 constexpr，此 for 循环在编译期展开，无运行时控制流
    for r in range(BLOCK_ROWS):
        row_idx = base_row + r
        if row_idx < n_rows:
            freq_row_idx = row_idx // n_heads

            x_ptr = X + row_idx * stride_x
            out_ptr = Out + row_idx * stride_out
            cos_ptr = cos + freq_row_idx * stride_freq
            sin_ptr = sin + freq_row_idx * stride_freq

            x_real = tl.load(x_ptr + col_offsets * 2, mask=col_mask, other=0.0)
            x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=col_mask, other=0.0)
            cos_even = tl.load(cos_ptr + col_offsets * 2, mask=col_mask, other=0.0)
            sin_odd = tl.load(sin_ptr + col_offsets * 2 + 1, mask=col_mask, other=0.0)

            out_even = x_real * cos_even - x_imag * sin_odd
            out_odd = x_real * sin_odd + x_imag * cos_even

            tl.store(out_ptr + col_offsets * 2, out_even, mask=col_mask)
            tl.store(out_ptr + col_offsets * 2 + 1, out_odd, mask=col_mask)
```

一次的执行时间缩短到5ms，此时生成时间从8分钟减少到5分钟。

+ 