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


## Depedencies

These dependecies are required for X86 and for Xavier.

```
$ sudo apt-get install -y ninja-build 
```

## Clang for the Host computer

In the host computer, follow these instructions to download Clang:

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/10.x
$ mkdir build; cd build
```

There is no actual constraint regarding the Clang version for X86. Version 10 is selected but it could also be version 11, for example. However, for Xavier there is a constraint, as explained next.


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


Clang for Xavier (aarch64) can be version 11, without applying any modification. 
On the other hand, if for any reason an older LLVM version is required, the modification is simple.

If one tries to compile LLVM version 10 or earlier for aarch64, the following message will appear hidden among 
tons of other messages:

```
Not building CUDA offloading plugin: only support CUDA in Linux x86_64 or ppc64le hosts
```

It turns out, by reading this [issue](https://reviews.llvm.org/D76469), that aarch64 was not tested for LLVM 10.
However, it works. The modification in the source code is straight forward. Just compare the following line of code 
in [LLVM version 11](https://github.com/llvm/llvm-project/blob/280e47ea0e837b809be03f2048ac8abc14dbc387/openmp/libomptarget/plugins/cuda/CMakeLists.txt#L12) and and [LLVM version 10](https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/libomptarget/plugins/cuda/CMakeLists.txt#L12). It is just a matter of
copying these two lines from LLVM version 11 to 10 to enable *libomptarget* compilation for LLVM 10.

Once this version issue is settled, follow these instructions to download Clang for LLVM version 10:

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/10.x
apply the patch as explained above
$ mkdir build; cd build
```

Or these instructions to download Clang for LLVM version 11, where no patch is required.

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/11.x
$ mkdir build; cd build
```

Next, it is the LLVM/Clang configuration itself. For the [NVidia Xaxier](https://releases.llvm.org/10.0.0/docs/HowToBuildOnARM.html) board, the configuration command is: 

```
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="ARM;AArch64;NVPTX" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a" ../llvm
```

Where the main difference is the ARM build targets *-DLLVM_TARGETS_TO_BUILD="ARM;AArch64;NVPTX"*
and the ARM CPU architecture *-DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a"* according to the Xavier spec.

Before running the actual compilation in Xavier, it is recommended to turn on the performance
mode. This way the 8 cores will be available to be used for the compilation, otherwise, only 4 cores are avaiçabçe for default.

Finally, run the compilation:

```
$ nohup nice -5 ninja
$ ninja install
```

*nohup* is recommend in case the terminal to Xavier is closed. *nice -5* is recommend to reduce the priority of the compilation since Ninja uses all available cores for default.

Once the compilation is done, set the *PATH* and *LD_LIBRARY_PATH*.
A first check is to run the following command to see the registered targets. We are expecting to find NVIDIA target for GPU offloading.

```
$ llc --version
LLVM (http://llvm.org/):
  LLVM version 10.0.1
  Optimized build.
  Default target: aarch64-unknown-linux-gnu
  Host CPU: (unknown)

  Registered Targets:
    aarch64    - AArch64 (little endian)
    aarch64_32 - AArch64 (little endian ILP32)
    aarch64_be - AArch64 (big endian)
    arm64      - ARM64 (little endian)
    arm64_32   - ARM64 (little endian ILP32)
    nvptx      - NVIDIA PTX 32-bit
    nvptx64    - NVIDIA PTX 64-bit
```


## Debugging libomptarget


--> You need to compile libomp with -DOMPTARGET_DEBUG so that debug output is enabled.
https://github.com/clang-ykt/clang/issues/14#issuecomment-301114816

## Clang for the Xavier with a remote disk

In case you were not able to compile Clang with the previous process, possibly because of lack of disk space, one alternative is to mount a bigger a remote disk at the expense of performance. 
To minimize the impact in performance, prefer to select a remote disk in the same network as Xavier.

Let's assume that the Clang source code is stored in *~/llvm-project* in the host computer
and we want to mount it in *~/mnt/llvm-project* in the Xavier board. In the Xavier board execute:

```
$ sudo apt-get install sshfs
$ sshfs <user>@<host-ip>:~/llvm-project ~/mnt/llvm-project
```

Another alternative is to mount the remote drive via [NFS](https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-18-04).

Then, proceed with the normal Clang configuration for Xavier. 

# Cross Compiling CLANG for Xavier

Another alternative is to perform cross compiling in the host computer. The advantage is the fastest compilation speed, but the initial compilation setup is a bit challenging, as explained next.

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

## Additional requirements

PkgConfig is used in LLVM. So it is necessary to install an [PkgConfig wrapper](https://autotools.io/pkgconfig/cross-compiling.html) to the host computer 
to enable cross compilation with PkgConfig.

```
$ sudo apt install pkg-config-aarch64-linux-gnu
```

## Cross compiling a simple application

For testing purposes, let's compile a simple application without any depedency or dynamic library.
A classic HelloWorld is good enough for testing purposes. Let's also use CMake and Ninja as build systems
for cross compilation.

```
```

## Cross compiling LLVM/Clang

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
cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=../Toolchain_aarch64_l4t.cmake  -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX" -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" -DLLVM_TABLEGEN="/usr/bin/llvm-tblgen-6.0" -DLLVM_TARGET_ARCH="AArch64" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 ../llvm
```

**TO BE COMPLETED*!!!!*


## Cross compiling an CUDA aplication

**TO BE COMPLETED*!!!!*

These are the requirements for the host, already explained in the previous sections:
 - The ARM cross-compilation targeting aarch64 architecture (e.g. gcc-aarch64-linux-gnu);
 - Mount the remote sysroot.


One additional requirement for the host when cross compiling CUDA is an adequate compiler.
In this case, follow [these instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#cross-platform) to cross compile for CUDA, and install it in the host:

```
$ sudo apt-get install cuda-cross-aarch64

```

It seems that *NVIDIA SDK Manager* also includes the cross compilers for cuda-cross-aarch64.
However, this was not tested yet. Please refer to these links for further information:

- See Section [1.2. NVIDIA SDK Manager](https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#package-management-tool);
- Installing [SDK Manager for Jetson](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html). In this step, the Xavier needs to be connected to the host computer.
I suppose that step 3 can be skipped assuming that the Xavier is already configured to run CUDA apps. You will also need the same version of CUDA that comes with the JetPack.

Next, the following [cross compilation procedure](https://docs.nvidia.com/vpi/sample_cross_aarch64.html) provided by NVidia is the easiest found so far. It only requires to setup a single CMake file, called *Toolchain_aarch64_l4t.cmake*, with the cross compile configuration.
This file might require some editing according to the installation location to the toolchain.

```
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
 
set(target_arch aarch64-linux-gnu)
set(CMAKE_LIBRARY_ARCHITECTURE ${target_arch} CACHE STRING "" FORCE)
 
# Configure cmake to look for libraries, include directories and
# packages inside the target root prefix.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH "/usr/${target_arch}")
 
# needed to avoid doing some more strict compiler checks that
# are failing when cross-compiling
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
 
# specify the toolchain programs
find_program(CMAKE_C_COMPILER ${target_arch}-gcc)
find_program(CMAKE_CXX_COMPILER ${target_arch}-g++)
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Can't find suitable C/C++ cross compiler for ${target_arch}")
endif()
 
set(CMAKE_AR ${target_arch}-ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ${target_arch}-ranlib)
set(CMAKE_LINKER ${target_arch}-ld)
 
# Not all shared libraries dependencies are instaled in host machine.
# Make sure linker doesn't complain.
set(CMAKE_EXE_LINKER_FLAGS_INIT -Wl,--allow-shlib-undefined)
 
# instruct nvcc to use our cross-compiler
set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)
```

Then, in the source dir of the application, execute:

```
cmake . -DCMAKE_TOOLCHAIN_FILE=Toolchain_aarch64_l4t.cmake
```


https://developer.nvidia.com/blog/building-cuda-applications-cmake/

#=========


https://developer.nvidia.com/embedded/linux-tegra

Install it in the Host Computer
https://developer.nvidia.com/nsight-compute





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

# Other sources for testing GPU offloading

- https://github.com/pc2/OMP-Offloading
- https://crpl.cis.udel.edu/ompvvsollve/results/
- https://github.com/SOLLVE/sollve_vv/blob/master/tests/4.5/target/test_target_map_array_default.c

# Info about compiling LLVM for ARM

- https://releases.llvm.org/10.0.0/docs/HowToBuildOnARM.html
- `Cross Compiling <https://releases.llvm.org/10.0.0/docs/HowToCrossCompileLLVM.html>`_ is probably not an easy option because the host wont have access to the GPU drivers specific for the Xavier board. Please let me know if you know how to do it !!!
