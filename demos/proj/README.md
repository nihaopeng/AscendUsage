# 工程化算子开发（支持GE入图）

详情参考[算子开发示例](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html)

## 工程创建

+ msopgen创建工程结构

首先编写算子json，add_custom.json：

```json
[
    {
        "op": "AddCustom",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND","ND","ND"],
                "type": ["fp16","float","int32"]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND","ND","ND"],
                "type": ["fp16","float","int32"]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": ["ND","ND","ND"],
                "type": ["fp16","float","int32"]
            }
        ]
    }
]
```

使用msopgen创建工程结构(文档基于cann 9.0的框架，注意8.5和9.0的框架结构和tiling data的定义方式都不相同)：

```bash
${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i $HOME/sample/add_custom.json -c ai_core-<soc_version> -lan cpp -out $HOME/sample/AddCustom
```

-c：ai_core-<soc_version>代表算子在AI Core上执行，<soc_version>为昇腾AI处理器的型号。

> 在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，获取Chip Name信息。实际配置值为AscendChip Name，例如Chip Name取值为xxxyy，实际配置值为Ascendxxxyy。

## 算子编写

在demos/proj/op_kernel/add_custom.cpp中编写算子实现代码。

```cpp

```

## 编译

如果使用虚拟环境中的cann

注意将CMakePresets.json中的ASCEND_CANN_PACKAGE_PATH修改

完成后执行

`./build.sh`

编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包custom_opp_<target os>_<target architecture>.run。

## 部署

算子包部署指执行自定义算子包的安装，算子工程的编译结果会自动部署到算子包安装目录下。

`./custom_opp_<target os>_<target architecture>.run --install-path=<path>`

--install-path为可选参数，用于指定自定义算子包的安装目录。支持指定绝对路径，运行用户需要对指定的安装路径有可读写权限。

> 如果部署算子包时通过配置--install-path参数指定了算子包的安装目录，则在使用自定义算子前，需要执行source <path>/vendors/<vendor_name>/bin/set_env.bash命令，set_env.bash脚本中将自定义算子包的安装路径追加到环境变量ASCEND_CUSTOM_OPP_PATH中，使自定义算子在当前环境中生效。

## 算子注册PyTorch

如果需要注册