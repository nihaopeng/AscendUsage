# 基础路径配置 (请根据实际环境检查 INSTALL_DIR)
INSTALL_DIR := /usr/local/Ascend/ascend-toolkit/latest
RT_INC      := $(INSTALL_DIR)/include
RT_LIB      := $(INSTALL_DIR)/lib64

# 获取依赖包的库 python+PyTorch+Torch_NPU 的编译参数 (核心修复点)
PYTHON_BIN  := python3
PY_DEV_INC  := $(shell $(PYTHON_BIN) -c "import sysconfig; print('-I ' + sysconfig.get_path('include'))")
PY_INC      := $(shell $(PYTHON_BIN) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")  # 获取 PyTorch 的 include 路径
NPU_INC     := $(shell $(PYTHON_BIN) -c "import os, torch_npu; print('-I ' + os.path.dirname(torch_npu.__file__) + '/include');")
PY_LIB_PATH := $(shell $(PYTHON_BIN) -c "import torch; print('-L ' + torch.utils.cmake_prefix_path + '/../../lib')")
NPU_LIB_PATH:= $(shell $(PYTHON_BIN) -c "import os, torch_npu; print('-L ' + os.path.dirname(torch_npu.__file__) + '/lib');")
TORCH_LIBS  := -ltorch -ltorch_cpu -lc10 -ltorch_python -ltorch_npu

# LIB_DIR
LIBS_DIR := -L$(RT_LIB) $(PY_LIB_PATH) $(NPU_LIB_PATH)

# LIB
LIBS := -lascendcl $(TORCH_LIBS)

# INC_DIR
INC_DIR := -I$(RT_INC) -I csrc/kernels -I $(RT_INC) $(PY_INC) $(NPU_INC) $(PY_DEV_INC)

# 目标文件定义
TARGET      := build/libdevice.so
DEVICE      := build/libmatmul_kernel.so

# FLAGS
CXXFLAGS    := -O2 -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 --shared -L. -Wl,-rpath,'$$ORIGIN'
BISHENGFLAGS := -O2 -fPIC --npu-arch=dav-2201 --cce-auto-sync --shared

# 5. 构建规则
all: $(TARGET)

$(TARGET):
	@echo "include flags: $(INC_DIR)"
	@echo "library dir flags: $(LIBS_DIR)"
	@echo "library flags: $(LIBS)"

	mkdir -p build
	@echo "正在编译 ASC 算子..."
	bisheng $(BISHENGFLAGS) \
	csrc/kernels/matmul.asc \
	-o $(DEVICE)
	
	@echo "正在编译 C++ 接口并链接生成最终库..."
	# 这里将 .asc 编译出的中间产物与 .cpp 一起链接成最终的 .so
	g++ $(CXXFLAGS) \
	csrc/ascend_interface.cpp csrc/kernels/matmul_torch.cpp $(DEVICE) \
	$(INC_DIR) \
	$(LIBS_DIR) \
	$(LIBS) \
	-o $(TARGET)
	
	cp build/libdevice.so .
	cp build/libadd_kernel.so .

clean:
	rm -rf build
	rm -f libdevice.so
	rm -f libadd_kernel.so
	rm -f $(TARGET)

.PHONY: all clean