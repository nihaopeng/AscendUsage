# Ascend Usage

华为昇腾（Ascend）AI处理器使用示例和算子开发项目。

## 项目结构

```
├── demos/                    # 示例代码
│   ├── framework/           # PyTorch自定义算子框架示例
│   │   ├── run_ops.py      # 向量加法算子性能测试
│   │   ├── README.md       # 安装和运行说明
│   │   └── build/          # 构建输出
│   ├── exe/                # 可执行文件示例
│   └── proj/               # CMake项目示例
├── operators/              # 自定义算子实现
│   └── gemm/              # 矩阵乘法算子
│       ├── csrc/          # C++源码
│       │   ├── ascend_interface.cpp
│       │   └── kernels/   # 内核实现
│       └── run_ops.py     # GEMM性能测试
├── docs/                   # 文档
│   └── README.md          # 昇腾问题收集
└── .vscode/               # VSCode配置
```

## 快速开始

### 环境准备

```bash
conda create -n AscendUsage python=3.11
conda activate AscendUsage

# 安装基础依赖
pip install PyYAML "numpy<2" "setuptools<70" scipy decorator attrs psutil synr

# 安装PyTorch (ARM64版本)
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 安装torch_npu (昇腾适配版本)
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post8-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 运行示例

#### 向量加法算子测试

```bash
cd demos/framework
make
python run_ops.py
```

#### 矩阵乘法算子测试

```bash
cd operators/gemm
# 需要先编译生成libdevice.so
python run_ops.py
```

## 算子开发

### 向量加法算子

- **位置**: `demos/framework/`
- **功能**: 比较Vector和Scalar版本的向量加法性能
- **特点**: 展示昇腾向量化指令的优势

### 矩阵乘法算子

- **位置**: `operators/gemm/`
- **功能**: 高性能矩阵乘法实现
- **特点**: 支持大矩阵运算，展示昇腾计算能力

## 性能测试

项目包含完整的性能测试框架，支持：
- 预热运行消除冷启动影响
- 异步算子同步确保计时准确
- 结果精度验证（相对误差和绝对误差）
- 性能加速比计算

## 文档

- [问题收集](./docs/README.md) - 昇腾使用过程中的问题和解决方案
- [算子开发指南](./demos/framework/README.md) - PyTorch自定义算子开发流程

## 注意事项

1. 确保昇腾驱动和CANN工具包已正确安装
2. 算子开发需要熟悉昇腾编程接口和硬件特性
3. 性能测试时注意数据对齐和内存访问模式
4. 精度验证时根据算子类型调整误差容忍度

## 贡献

欢迎提交Issue和Pull Request，共同完善昇腾使用示例。

## 许可证

MIT License