**注：已经都安装好了**

针对你的 C++20 + CUDA C++ 音频 AI 训练/推理框架，依赖要分别安装在两台机器上：

| 机器               | 系统         | 安装重点                                   |
| ---------------- | ---------- | -------------------------------------- |
| 本地 i5-13420H     | Arch Linux | oneDNN、oneMKL、OpenVINO、音频 DSP 库        |
| 远端 RTX 3060 12GB | Ubuntu     | CUDA 12.4、cuBLASLt、cuFFT、cuDNN、CUTLASS |

建议先装底层计算库，把 LibTorch 和 TensorRT 作为可选组件，避免第一版引入过多依赖。

## 1. 本地 Arch Linux：CPU 开发环境

Arch 官方仓库已经提供了 oneDNN、Intel oneMKL、OpenVINO，所以不需要手动从源码编译。

[image](https://www.google.com/s2/favicons?domain=https://archlinux.org\&sz=32)

onednn 3.11.3-1 (x86_64)

+2



直接执行：

```
sudo pacman -Syu
sudo pacman -S --needed \
    base-devel cmake ninja git pkgconf \
    gdb clang lld \
    onednn intel-oneapi-mkl openvino \
    fftw libsndfile libsoxr
```

这些库分别负责：

| 软件包                | 作用               |
| ------------------ | ---------------- |
| `onednn`           | 神经网络前向、反向算子      |
| `intel-oneapi-mkl` | GEMM、BLAS、FFT    |
| `openvino`         | Intel CPU 推理性能基线 |
| `fftw`             | FFT 对照测试         |
| `libsndfile`       | WAV、FLAC 音频文件读写  |
| `libsoxr`          | 高质量重采样           |
| `clang` / `lld`    | 可选 C++ 编译和链接工具   |

Arch 的 `libsoxr` 与 `libsndfile` 软件包包含开发所需的库与头文件。

[image](https://www.google.com/s2/favicons?domain=https://archlinux.org\&sz=32)

libsoxr 0.1.3-5 (x86_64)

+1



### 检查安装

```
pacman -Q onednn intel-oneapi-mkl openvino fftw libsndfile libsoxr

ls /opt/intel/oneapi/mkl/latest/include/mkl.h
ls /usr/include/oneapi/dnnl/dnnl.hpp
ls /usr/include/sndfile.h
```

MKL 的头文件在 `/opt/intel/oneapi/mkl/` 下；系统库通常安装在 `/usr/lib`。

可以额外检查 MKL 的 CMake 配置：

```
find /opt/intel/oneapi/mkl -name 'MKLConfig.cmake'
```

Intel IPP 是可选项，先不必安装。你已经有 oneMKL FFT 和 FFTW，足以开展音频频谱计算性能测试；之后确认 IPP 在具体处理场景中有收益，再安装 Intel 官方 IPP SDK。

## 2. 远端 Ubuntu NAS：RTX 3060 / CUDA 12.4

### 第一步：安装 C++ 开发与音频库

在 NAS 上执行：

```
sudo apt update
sudo apt install -y \
    build-essential cmake ninja-build git pkg-config \
    libsndfile1-dev libsoxr-dev libfftw3-dev \
    zlib1g-dev
```

### 第二步：检查已有的 CUDA 库

你已经安装了 CUDA Toolkit 12.4，先不要重复安装。

```
nvcc --version
nvidia-smi

ls /usr/include/cuda_runtime.h
ls /usr/include/cublasLt.h
ls /usr/include/cufft.h

ldconfig -p | grep -E 'lib(cudart|cublasLt|cufft)\.so'
```

如果这些头文件和动态库存在，cuBLASLt、cuFFT、CUDA Runtime 就已经可以用于 C++ 开发，不需要再分别安装。

如果头文件缺失，先检查 Ubuntu 的 CUDA 开发包：

```
dpkg -l | grep -E 'nvidia-cuda-toolkit|nvidia-cuda-dev'
```

只有确认对应开发包缺失后，再考虑安装 `nvidia-cuda-dev` 或 NVIDIA 官方的 `cuda-libraries-dev-12-4`。不要混装多个 CUDA 版本的头文件和库。

### 第三步：安装 cuDNN 9

cuDNN 通常需要单独安装。

先检查：

```
ldconfig -p | grep libcudnn
dpkg -l | grep cudnn
apt-cache policy cudnn9-cuda-12
```

如果 NVIDIA CUDA 软件源已经启用，可以先模拟安装：

```
sudo apt-get -s install cudnn9-cuda-12
```

确认输出没有提出卸载现有 CUDA 包或更换驱动后，再执行：

```
sudo apt install cudnn9-cuda-12
```

这是 NVIDIA 官方推荐的 CUDA 12 对应 cuDNN 9 安装方式。

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NVIDIA cuDNN Installation



如果提示找不到软件包，需要先配置 NVIDIA 官方软件源。以下适用于官方支持的 Ubuntu x86-64 版本：

```
distro="ubuntu$(. /etc/os-release; echo "$VERSION_ID" | tr -d '.')"

wget "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

然后重新检查并安装 `cudnn9-cuda-12`。如果该仓库路径不可用，以 [NVIDIA cuDNN 官方安装指南](https://docs.nvidia.com/cudnn/latest/installation/backend/linux/?utm_source=chatgpt.com) 为准。

注意：NAS 已有 595.71.05 驱动，不要执行 `sudo apt install cuda` 或 `cuda-drivers`。 这些元包可能改变驱动和默认 CUDA 配置，而 `cuda-toolkit-12-4` 是不包含驱动的版本化开发工具包。

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

Installation Guide for Linux 12.4 documentation



### 第四步：安装 CUTLASS

CUTLASS 是 Header-only C++ 模板库，直接克隆就能作为依赖使用。

```
mkdir -p ~/deps
cd ~/deps

git clone --depth 1 https://github.com/NVIDIA/cutlass.git
```

使用时，在 CMake 添加：

```
target_include_directories(your_target PRIVATE
    "$ENV{HOME}/deps/cutlass/include"
)
```

如果要编译 CUTLASS 的测试或 Profiler，建议明确限制目标架构为 RTX 3060 的 SM 8.6：

```
cmake -S ~/deps/cutlass -B ~/deps/cutlass/build \
    -DCUTLASS_NVCC_ARCHS=86 \
    -DCUTLASS_ENABLE_TESTS=OFF
```

CUTLASS 官方确认支持 CUDA 12.x 和 Ampere，作为 Header-only 库也不必编译整个项目才能使用。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 3. 用 CMake 管理依赖

建议不要手动添加大量 `-I`、`-L` 和 `-l` 参数，而是使用 CMake 的 imported targets。

例如，GPU 项目可以从下面这个 `CMakeLists.txt` 开始：

```
cmake_minimum_required(VERSION 3.24)
project(audio_ai_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_executable(audio_ai main.cpp)

target_link_libraries(audio_ai PRIVATE
    CUDA::cudart
    CUDA::cublasLt
    CUDA::cufft
)
```

`CUDA::cublasLt`、`CUDA::cufft` 等都是 CMake 官方支持的 CUDA imported targets。

[image](https://www.google.com/s2/favicons?domain=https://cmake.org\&sz=32)

CMake 4.5.0-rc1 Documentation



构建：

```
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

由于 NAS 的 CUDA 12.4 是通过 Ubuntu 发行版包安装的，没有 `/usr/local/cuda`，CMake 应优先从已有的 `/usr/bin/nvcc` 检测 Toolkit。如果无法检测，再根据实际安装路径指定 `CUDAToolkit_ROOT` 或 `CMAKE_CUDA_COMPILER`，不要假设 `/usr/local/cuda` 存在。

## 4. LibTorch、TensorRT 和 IPP 先不装也行

这三个可以留到后面：

| 库         | 何时需要安装                    | 来源                                                                                                         |
| --------- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| LibTorch  | 需要 PyTorch C++ 自动求导、训练对照时 | [PyTorch 官方](https://pytorch.org/get-started/locally/?utm_source=chatgpt.com)                              |
| TensorRT  | 完整 GPU 推理引擎性能对照时          | [NVIDIA 官方](https://developer.nvidia.com/tensorrt?utm_source=chatgpt.com)                                  |
| Intel IPP | CPU 音频 DSP、重采样、FFT 优化时    | [Intel 官方](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html?utm_source=chatgpt.com) |

LibTorch 需要根据 CPU/CUDA 构建版本下载对应二进制发行包，不能随意选一个最新的 GPU 包就假定与 CUDA 12.4 完全匹配。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch main documentation



## 5. 安装完成后最终应该是这样

本地 Intel CPU

Arch Linux

oneDNN · oneMKL · OpenVINO · FFTW · libsndfile · libsoxr

远端 NVIDIA RTX 3060

Ubuntu / CUDA Toolkit 12.4

CUDA Runtime · cuBLASLt · cuFFT · cuDNN 9 · CUTLASS

建议第一版只依赖这些。这样你就可以直接开始编写 CPU AVX2、oneDNN、CUDA GEMM、cuFFT 和音频模型的 C++ 算子，后续再逐步加入训练自动求导和推理编译器。

特别提醒：oneMKL 的安装体积较大，Arch 软件包安装后约占 1.6GB，而 CUDA/cuDNN 也可能占用较多磁盘空间；NAS 上安装前建议检查剩余容量。

[image](https://www.google.com/s2/favicons?domain=https://archlinux.org\&sz=32)

intel-oneapi-mkl 2026.0.0_908-1 (x86_64)
