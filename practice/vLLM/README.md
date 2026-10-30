# vLLM run on ascend

## install

uv venv --seed --python 3.12

source .venv/bin/activate

uv pip install vllm==0.23.0

uv pip install torch-npu=2.10.0.post4

uv pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple vllm-ascend==0.23.0
