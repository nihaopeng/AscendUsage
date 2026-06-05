# comfyUI usage

## env

`conda create -n comfyui python=3.11`

`conda activate comfyui`

`git clone https://github.com/comfyanonymous/ComfyUI.git`

`cd ComfyUI`

`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

+ verify

`python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"`

`pip3 install torchvision==0.20.1 torch==2.5.1 torch_npu==2.5.1 decorator -i https://pypi.tuna.tsinghua.edu.cn/simple`

`pip install torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu`

`pip install -U --pre comfyui-manager`

`pip install matrix-nio`

``

```bash
cd ComfyUI/
vim main.py
# 添加以下内容
import torch 
import torch_npu
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
```

## startup

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python main.py --listen 0.0.0.0 --port 59075 --enable-manager`

## config

```bash
cp extra_model_paths.yaml.example extra_model_paths.yaml
mkdir -p /mnt/nvme1p1/pengyt/comfyui/custom_nodes

vim extra_model_paths.yaml
```

```yaml
other_ui:
   base_path: /mnt/nvme1p1/pengyt/comfyui
   checkpoints: models/checkpoints
   gligen: models/gligen
   custom_nodes: /mnt/nvme1p1/pengyt/comfyui/custom_nodes
```

## run

`pip install "huggingface_hub[cli]"`

download model to path:

`hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged --local-dir /mnt/nvme1p1/pengyt/comfyui/models/checkpoints`