cmake_minimum_required(VERSION 3.16)

# required find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
find_package(ASC REQUIRED)

# required 指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译，异构代码后缀一定需要是asc
project(kernel_samples LANGUAGES ASC CXX)

# 编译异构代码为 动态 库
add_library(kernel_lib SHARED src/device.asc src/interface.asc)

# required
add_executable(demo src/main.cpp)

# required 指定include
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest/")
set(INCLUDE_DIRS ${ASCEND_PATH}/include ${CMAKE_SOURCE_DIR}/include)
target_include_directories(demo PRIVATE ${INCLUDE_DIRS})
target_include_directories(kernel_lib PRIVATE ${INCLUDE_DIRS})

# 指定链接库
# target_link_libraries(kernel_lib PRIVATE ascendcl)
target_link_libraries(demo PRIVATE kernel_lib)

# required 通过编译选项设置NPU架构
set(ASC_OPTIONS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
target_compile_options(kernel_lib PRIVATE ${ASC_OPTIONS})