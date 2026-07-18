# Flux T2I / I2I 执行流程分析

> CANN 9.0.0 / torch_npu 2.6.0 / Ascend 910B3

---

## 1. 入口分发

**文件**: `flux/src/flux/__main__.py`

| 命令 | 函数 | 文件 |
|------|------|------|
| `python -m flux t2i ...` | `cli_main()` | `flux/cli.py` |
| `python -m flux kontext ...` | `kontext_main()` | `flux/cli_kontext.py` |
| `python -m flux control ...` | `control_main()` | `flux/cli_control.py` |
| `python -m flux fill ...` | `fill_main()` | `flux/cli_fill.py` |
| `python -m flux redux ...` | `redux_main()` | `flux/cli_redux.py` |

```python
Fire({"t2i": cli_main, "kontext": kontext_main, "control": control_main, ...})
```

---

## 2. T2I 完整流程

### 2.1 入口

**文件**: `flux/src/flux/cli.py`
**函数**: `main(name, width, height, seed, prompt, device, num_steps, guidance, offload, ...)`

### 2.2 模型加载

| 加载函数 | 文件 | 模型 | 输出形状 |
|---------|------|------|---------|
| `load_t5(device, max_length)` | `util.py` | `google/t5-v1_1-xxl` → `HFEmbedder` | `(B, 512, 4096)` |
| `load_clip(device)` | `util.py` | `openai/clip-vit-large-patch14` → `HFEmbedder` | `(B, 768)` |
| `load_flow_model(name, device)` | `util.py` | `Flux` / `FluxLoraWrapper` | 12B 参数, ~24GB |
| `load_ae(name, device)` | `util.py` | `AutoEncoder` | VAE, 16ch latent |

`load_flow_model` 内部流程:
1. `torch.device("meta")` 创建空壳模型
2. `load_sft_with_tqdm()` 从 safetensors 逐张量加载到 CPU
3. `optionally_expand_state_dict()` 处理形状不匹配
4. `load_state_dict(sd, strict=False, assign=True)` 赋值到模型
5. `_move_model_params_to_npu()` flat buffer 单次搬运到 NPU

### 2.3 噪声生成

**文件**: `flux/src/flux/sampling.py`
**函数**: `get_noise(num_samples, height, width, device, dtype, seed)`

> 输出形状: `(1, 16, 2*ceil(H/16), 2*ceil(W/16))` (2x 来自 patch packing)

### 2.4 条件准备

**文件**: `flux/src/flux/sampling.py`
**函数**: `prepare(t5, clip, img, prompt)`

```
prompt ─┬→ t5(prompt)     → txt:     (1, seq_len, 4096)    # HFEmbedder.forward()
        └→ clip(prompt)    → vec:     (1, 768)              # HFEmbedder.forward()

noise ──→ einops.rearrange → img:     (1, num_patches, 64)  # 2x2 pack
         position ids       → img_ids: (1, num_patches, 3)   # [0, y_pos, x_pos]
         text position ids  → txt_ids: (1, seq_len, 3)       # 全零
```

### 2.5 时间步调度

**文件**: `flux/src/flux/sampling.py`
**函数**: `get_schedule(num_steps, image_seq_len, shift)`

- dev 模型: 带 shift (`base_shift=0.5` → `max_shift=1.15` 按序列长度插值)
- schnell 模型: 均匀时间步

### 2.6 去噪循环

**文件**: `flux/src/flux/sampling.py`
**函数**: `denoise(model, img, img_ids, txt, txt_ids, vec, timesteps, guidance, img_cond=None, img_cond_seq=None, img_cond_seq_ids=None)`

```python
# 整流流 (Rectified Flow) + 欧拉积分
for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                 y=vec, timesteps=t_vec, guidance=g_vec)
    img = img + (t_prev - t_curr) * pred    # x_{t-1} = x_t + Δt * v_θ
```

### 2.7 模型前向

**文件**: `flux/src/flux/model.py`
**类/方法**: `Flux.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance)`

```
img (N,64) ──→ img_in         → (N, 3072)
timesteps   ──→ time_in       ┐
guidance    ──→ guidance_in   ├→ vec (3072)   # adaLN 调制
y (CLIP)    ──→ vector_in     ┘
txt (L,4096)──→ txt_in        → (L, 3072)

ids = cat(txt_ids, img_ids) ──→ pe_embedder  → RoPE 位置编码

┌─ 19× DoubleStreamBlock ──────────────────────────┐
│  img/txt 各独立 attn + mlp, adaLN 调制           │
└──────────────────────────────────────────────────┘

img = cat(txt, img) ──→ 38× SingleStreamBlock ──→ img[:, txt_len:, :]
                        统一 attn + mlp

img ──→ final_layer(img, vec) → (N, 64)   # 速度预测
```

**关键子模块** (`flux/src/flux/modules/layers.py`):

| 类 | 数量 | 用途 |
|----|------|------|
| `DoubleStreamBlock` | 19 | 图像/文本双流注意力 + 各自 MLP |
| `SingleStreamBlock` | 38 | 统一序列注意力 + MLP |
| `LastLayer` | 1 | adaLN + 线性投影 3072→64 |
| `EmbedND` | 1 | N 维 RoPE (axes_dim=[16,56,56]) |

### 2.8 VAE 解码

**文件**: `flux/src/flux/sampling.py`, `flux/src/flux/modules/autoencoder.py`
**函数**: `unpack(x, height, width)`, `ae.decode(x)`

```python
x = unpack(x, h, w)    # (N, 64) → (1, 16, H/8, W/8)  反转 pack
x = ae.decode(x)         # (1, 16, H/8, W/8) → (1, 3, H, W)
```

VAE 内部: `z = z / 0.3611 + 0.1159` → 残差块 → 上采样 → RGB

### 2.9 图像保存

**文件**: `flux/src/flux/util.py`
**函数**: `save_image(nsfw_classifier, name, output_name, idx, x, ...)`, `embed_watermark()`

```
clamp[-1,1] → watermark (DWT-DCT) → PIL Image → NSFW 检测 → JPEG + EXIF
```

---

## 3. I2I / Kontext 完整流程

### 3.1 入口

**文件**: `flux/src/flux/cli_kontext.py`
**函数**: `main(name, aspect_ratio, seed, prompt, device, num_steps, guidance, img_cond_path, ...)`

> 与 T2I 区别: 固定 `name="flux-dev-kontext"`, 默认 `device="npu"`, 加载 `PixtralContentFilter`

### 3.2 内容过滤

**文件**: `flux/src/flux/content_filters.py`
**类**: `PixtralContentFilter`

```python
cf = PixtralContentFilter(torch.device("cpu"))
cf.test_txt(prompt)    # FalconAI NSFW + Pixtral-12B 版权检测
cf.test_image(input)   # 同上
```

### 3.3 输入准备

**文件**: `flux/src/flux/sampling.py`
**函数**: `prepare_kontext(t5, clip, prompt, ae, img_cond_path, target_width, target_height, bs, seed, device)`

```
输入图像 ─→ resize (PREFERED_KONTEXT_RESOLUTIONS 最佳匹配)
         ─→ ae.encode(img_cond)       → img_cond_seq:     (1, N_cond, 64)
         ─→ position ids (axis_0=1)   → img_cond_seq_ids: (1, N_cond, 3)

目标噪声 ─→ get_noise(target_res)     → (1, 16, ...)
prompt  ─→ prepare(...)               → {img, img_ids, txt, txt_ids, vec}

返回: {img, img_ids, txt, txt_ids, vec, img_cond_seq, img_cond_seq_ids}
```

**关键差异**: 条件图像通过 **序列维度** 拼接到噪声 token 后面（非通道维度）。`img_cond_seq_ids` 的 `axis_0=1` 标记条件 token。

### 3.4 去噪（Kontext 变体）

**文件**: `flux/src/flux/sampling.py`
**函数**: `denoise()` (同 T2I, 额外传入 `img_cond_seq`, `img_cond_seq_ids`)

```python
img_input = torch.cat((img, img_cond_seq), dim=1)         # 序列拼接
img_input_ids = torch.cat((img_ids, img_cond_seq_ids), dim=1)

pred = model(img=img_input, img_ids=img_input_ids, ...)

pred = pred[:, :img.shape[1]]   # 丢弃条件 token 的预测, 仅保留生成部分
img = img + (t_prev - t_curr) * pred
```

### 3.5 后处理

与 T2I 相同: `unpack()` → `ae.decode()` → 内容过滤 → `save_image(None, ...)`

---

## 4. 关键文件索引

| 文件 | 关键函数/类 | 用途 |
|------|------------|------|
| `flux/__main__.py` | `Fire({...})` | CLI 命令分发 |
| `flux/cli.py` | `main()` | T2I 主流程 |
| `flux/cli_kontext.py` | `main()` | I2I/Kontext 主流程 |
| `flux/cli_control.py` | `main()` | Canny/Depth 控制 |
| `flux/cli_fill.py` | `main()` | Inpainting |
| `flux/cli_redux.py` | `main()` | 图像变体 |
| `flux/sampling.py` | `get_noise()`, `prepare()`, `prepare_kontext()`, `denoise()`, `unpack()`, `get_schedule()` | 采样管线 |
| `flux/model.py` | `Flux`, `FluxLoraWrapper`, `FluxParams` | Transformer |
| `flux/util.py` | `load_flow_model()`, `load_ae()`, `load_t5()`, `load_clip()`, `_move_model_params_to_npu()`, `save_image()`, `load_sft_with_tqdm()` | 模型加载 |
| `flux/math.py` | `attention()`, `apply_rope()` | 注意力 & RoPE |
| `flux/modules/layers.py` | `DoubleStreamBlock`, `SingleStreamBlock`, `LastLayer`, `EmbedND`, `MLPEmbedder` | Transformer 层 |
| `flux/modules/autoencoder.py` | `AutoEncoder`, `Encoder`, `Decoder` | VAE |
| `flux/modules/conditioner.py` | `HFEmbedder` | T5/CLIP 封装 |
| `flux/modules/image_embedders.py` | `ReduxImageEncoder`, `CannyImageEncoder`, `DepthImageEncoder` | 条件编码器 |
| `flux/modules/lora.py` | `LinearLora`, `replace_linear_with_lora()` | LoRA |
| `flux/content_filters.py` | `PixtralContentFilter` | 内容安全 |
| `flux/trt/trt_manager.py` | `TRTManager` | TensorRT 推理 |

---

## 5. T2I vs I2I 核心差异

| | T2I | I2I (Kontext) |
|---|---|---|
| 输入 | 仅 prompt | prompt + 条件图像 |
| 条件注入方式 | — | 序列拼接 (`img_cond_seq`) |
| Position ID axis_0 | 全 0 | 条件=1, 生成=0 |
| `in_channels` | 64 | 64 |
| VAE 用途 | 仅 decode | encode 条件 + decode 输出 |

---

## 6. NPU 适配要点

| 问题 | 方案 | 位置 |
|------|------|------|
| `model.npu()` 死锁 (CANN 9.0.0-beta.1) | Flat buffer 单次搬运 | `util.py:_move_model_params_to_npu()` |
| OpenBLAS 线程超限 | 导入前设 `OPENBLAS_NUM_THREADS=1` | `__main__.py` |
| safetensors 不支持 `device="npu"` | CPU 加载后搬运 | `util.py:load_flow_model()` |

---

## 7. 安装与运行

### 可以去掉审查部分的代码
删除 PixtralContentFilter 的 import
删除 content_filter = PixtralContentFilter(torch.device("cpu")) 初始化
删除输入 prompt 和图片的审核检查（test_txt + test_image）
删除输出图片的审核检查

```bash
# 安装
conda create -n DiTs python=3.11
conda activate DiTs
pip install torch-2.6.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
pip install torch_npu-2.6.0.post5-cp311-cp311-manylinux_2_28_aarch64.whl
pip install -r requirements.txt

# 下载模型
hf download black-forest-labs/FLUX.1-Kontext-dev --local-dir checkpoints/black-forest-labs_FLUX.1-Kontext-dev

# 运行 T2I
cd /home/pengyt/projects/AscendUsage/practice/Flux1_dev/flux/src && conda activate DiTs && FLUX_MODEL='/mnt/nvme1p1/pengyt/Flux/black-forest-labs_FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors' FLUX_AE='/mnt/nvme1p1/pengyt/Flux/black-forest-labs_FLUX.1-Kontext-dev/ae.safetensors' ASCEND_RT_VISIBLE_DEVICES=2 python -m flux t2i --prompt "a cat in a hat" --width 1024 --height 1024

# 运行 I2I (Kontext)
cd /home/pengyt/projects/AscendUsage/practice/Flux1_dev/flux/src && conda activate DiTs &&FLUX_MODEL='/mnt/nvme1p1/pengyt/Flux/black-forest-labs_FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors' FLUX_AE='/mnt/nvme1p1/pengyt/Flux/black-forest-labs_FLUX.1-Kontext-dev/ae.safetensors' ASCEND_RT_VISIBLE_DEVICES=2 python -m flux kontext --prompt "replace the logo with 'Hello'" --img_cond_path assets/grid.jpg
```
