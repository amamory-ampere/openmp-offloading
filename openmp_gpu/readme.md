# Introduction

This document explains how to setup OpenMP for GPU offloading in both X86 computer (refered as host computer) and in a NVidia Xavier board.

# Hardware requirements

In the host computer, it's necessary to check the *Computing Capability* of the existing GPU.
In the [clang/CmakeList.txt](https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/clang/CMakeLists.txt#L297) one will find 
these comments that say that the minimun hardware requiment for GPU offloading with OpenMP is 
a GPU with *Computing Capability 3.5*, refered as *sm_35*.  Please check [wikipedia](https://en.wikipedia.org/wiki/CUDA) to see the CUDA compute capability of your GPU.

```
# OpenMP offloading requires at least sm_35 because we use shuffle instructions
# to generate efficient code for reductions and the atomicMax instruction on
# 64-bit integers in the implementation of conditional lastprivate.
set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH "sm_35" CACHE STRING
  "Default architecture for OpenMP offloading to Nvidia GPUs.")
string(REGEX MATCH "^sm_([0-9]+)$" MATCHED_ARCH "${CLANG_OPENMP_NVPTX_DEFAULT_ARCH}")
if (NOT DEFINED MATCHED_ARCH OR "${CMAKE_MATCH_1}" LESS 35)
  message(WARNING "Resetting default architecture for OpenMP offloading to Nvidia GPUs to sm_35")
  set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH "sm_35" CACHE STRING
    "Default architecture for OpenMP offloading to Nvidia GPUs." FORCE)
endif()
```

The Xavier board has [Computing Capability 7.2](https://www.techpowerup.com/gpu-specs/jetson-agx-xavier-gpu.c3232). So it is compatible with OpenMP GPU offloading.

# Driver requirements

The CUDA environment needs to be installed first in both the host computer and in the Xavier board.
Please refer to instructions for the [Ubuntu 18.04 (host)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) and for [Xavier](https://developer.nvidia.com/embedded/jetpack).

# Setting up the CLANG compiler

Before starting the compilation, make sure that the computer has about 10GBytes of free disk space.
This is specially important for the Xavier board, with only 32GBytes of disk. The configuration presented here used 800MB of disk at the end of the compilation, but apparently it uses more
disk space in the middle of the compilation process. Due to this issue, it is highly recommended
to install a second disk. There is a nice tutorial about this in [JetsonHacks](https://www.jetsonhacks.com/2018/10/18/install-nvme-ssd-on-nvidia-jetson-agx-developer-kit/).

## Downloading Clang

 In both computers, follow these instructions to download Clang:

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/10.x
$ mkdir build; cd build
```

## Clang for the Host computer

Next, let's configure the CMAKE building system, with option for OpenMP offloading:

```
$ cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ../llvm
$ ninja
```

It is higly recommended to **use ninja build system instead of make**. The generated Makefile
has some sort of weird bug that, when you type 'make install', it compiles all over again... :/

The options *-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"* and *-DCMAKE_BUILD_TYPE=Release* are important to reduce compilation time. The parameter *LLVM_TARGETS_TO_BUILD* compile only for the required platforms. The argument *-DBUILD_SHARED_LIB=ON* is a good idea if the computer has less than 16GBytes of RAM. In a PC, the compilation time is about 35 min using **make -j 8**.

There are other LLVM projects that might be of interest. So, enable them individually as in this example:

```
$ CMAKE .... -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" ...
```

## Clang for the Xavier

For the [NVidia Xaxier](https://releases.llvm.org/10.0.0/docs/HowToBuildOnARM.html) board,
the configuration command is: 

```
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="ARM;AArch64;NVPTX" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a" ../llvm
```

Where the main difference is the ARM build targets *-DLLVM_TARGETS_TO_BUILD="ARM;AArch64;NVPTX"*
and the ARM CPU architecture *-DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a"* according to the Xavier spec.

# Cross Compiling CLANG for Xavier

If you were not able to compile Clang with the previous process, possibly because of lack of disk space, one alternative is to mount a bigger a remote disk at the expense of performance. Another
alternative is to perform cross compiling in the host computer. The advantage is fastest compilation speed, but the initial compilation setup is a bit challenging, as explained next: 

## Mounting the sysroot of the remote filesystem

Before starting to configure the compiler for cross compiling, it is necessary to 
configure the sysroot of the remote computer, in this case Xavier. We are going to 
mount the / from the Xavier to any directory of the host computer using [sshfs](https://wiki.dlang.org/GDC/Cross_Compiler/Existing_Sysroot).

I personaly prefer to insert these two commands in the *~/.bashrc* to ease mounting and unmounting the remote target. Then, just type these aliases to mount/unmount the remote target.

```
alias mount_xavier="sshfs <username>@<target_IP>:/ ~/mnt_xavier"
alias unmount_xavier="fusermount -u ~/mnt_xavier"
```
## Installing the remote toolchain

Next, we have to install the ARM compilers for Xavier in the host computer. These instructions are availabe in the [Jetson website](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fxavier_toolchain.html%23) and consist of the following step in the host computer:

```
$ mkdir $HOME/l4t-gcc
$ cd $HOME/l4t-gcc
$ tar xf gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
$ export CROSS_COMPILE=$HOME/l4t-gcc/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-
```

Note that we have installed only the ARM cross compilation toolchain. *It won't allow to cross compile CUDA applications*. For that, it is necessary to install the CUDA cross compiler toolchain, presented in the next sections.

## Downloading LLVM/Clang

The next step is to download Clang in the host computer:

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/10.x
# mkdir build; cd build
```

Then we get to the actual Clang configuration part. The key documents to setup Clang cross compilation are:
 - https://releases.llvm.org/10.0.0/docs/HowToCrossCompileLLVM.html
 - https://releases.llvm.org/10.0.0/docs/CMake.html

The later document describes the role of each CMAKE variable, including: 

```
LLVM_TARGET_ARCH:STRING
    LLVM target to use for native code generation. This is required for JIT generation. It defaults to “host”, meaning that it shall pick the architecture of the machine where LLVM is being built. If you are cross-compiling, set it to the target architecture name.
```

So, one has to set this variable according to Xavier. 

```
cmake -G Ninja -DCMAKE_CROSSCOMPILING=True -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="ARM;AArch64;NVPTX" -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" -DLLVM_TABLEGEN="/usr/bin/llvm-tblgen-10" -DCLANG_TABLEGEN="" -DLLVM_TARGET_ARCH="ARM" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a" ../llvm
```

**TO BE COMPLETED*!!!!*


## Cross compiling an CUDA aplication

**TO BE COMPLETED*!!!!*

https://developer.nvidia.com/embedded/linux-tegra

Install it in the Host Computer
https://developer.nvidia.com/nsight-compute

See '1.2. NVIDIA SDK Manager' in
https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#package-management-tool

In this step, the Xavier needs to be connected to the host computer.
https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html
I suppose that step 3 can be skipped because we already have the target configured.


$ sudo apt-get install cuda-cross-aarch64
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#cross-platform

## Cross compiling an OpenMP aplication

**TO BE COMPLETED*!!!!*

# Compiling the OpenMp application

Make sure that both CUDA and clang are with their required environment variables set. For example:

```
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib
$ export PATH=/usr/local/cuda/bin:$PATH
$ export CUDA_HOME=/usr/local/cuda          
```

Then, compile the OpenMP code with target:

```
$ cd /tmp
$ clang -v -pedantic -Wall -o omp-ser-cl omp-ser.c -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_35 -lc -lcudart  -ldl -L/usr/local/cuda/lib64 --cuda-path=/usr/local/cuda
```

The parameter *-march=sm_35* depends on the CPU model. Please check [wikipedia](https://en.wikipedia.org/wiki/CUDA) to see the CUDA compute capability of your GPU.

Then, type this commad to check if the dynamic libraries where found.

```
$ cd /tmp
$ ldd ./omp-ser-cl 
	linux-vdso.so.1 (0x00007ffde05cf000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007ff6b5e48000)
	libcudart.so.10.2 => /usr/local/cuda/lib64/libcudart.so.10.2 (0x00007ff6b5bca000)
	libomp.so => /usr/local/lib/libomp.so (0x00007ff6b5901000)
	libomptarget.so => /usr/local/lib/libomptarget.so (0x00007ff6b56f2000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007ff6b56d1000)
	/lib64/ld-linux-x86-64.so.2 (0x00007ff6b600f000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007ff6b56cc000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007ff6b56c0000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007ff6b553c000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007ff6b53b9000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007ff6b539f000)
```

# Back to the host again

In the host, in the directory *openmp-offloading/openmp_gpu*, execute:

```
$ ./omp-ser-cl
```

It is better to check if the GPU is working using [nsight-sys](https://developer.nvidia.com/nsight-systems):

```
$ /usr/local/cuda/nsight-sys
```

Then run the executable from nsight-sys to see the created threads and the GPU access.

# Other sources for tesing GPU offloading

- https://github.com/pc2/OMP-Offloading
- https://crpl.cis.udel.edu/ompvvsollve/results/
- https://github.com/SOLLVE/sollve_vv/blob/master/tests/4.5/target/test_target_map_array_default.c

# Info about compiling LLVM for ARM

- https://releases.llvm.org/10.0.0/docs/HowToBuildOnARM.html
- `Cross Compiling <https://releases.llvm.org/10.0.0/docs/HowToCrossCompileLLVM.html>`_ is probably not an easy option because the host wont have access to the GPU drivers specific for the Xavier board. Please let me know if you know how to do it !!!
