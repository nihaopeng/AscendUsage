# torch self-definition operator demo

## install

`conda create -n AscendUsage python=3.11`

`conda activate AscendUsage`

`pip install PyYAML`

`pip install "numpy<2"`

`pip install "setuptools<70"`

`pip install scipy decorator attrs psutil synr`

+ torch
`wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

`pip3 install torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

+ torch_npu

`wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

`pip3 install torch_npu-2.1.0.post8-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

`make`

## run

`python run_ops.py`