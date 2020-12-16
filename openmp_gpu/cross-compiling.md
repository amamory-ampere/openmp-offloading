


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