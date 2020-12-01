

# Hardware requirements

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

# Installing the CLANG compiler

The pre-built compiler is available [here](https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.0). I have not tested them yet.

To [compile Clang](https://freecompilercamp.org/llvm-openmp-build/), follow these instructions.

```
cmake -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_30 -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=30,35 -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ $LLVM_SRC/llvm
```

The parameter *sm_30* depends on the CPU model. In my case I use a Quadro K4000 GPU which has
CUDA compute capability 3.0. Please check [wikipedia](https://en.wikipedia.org/wiki/CUDA) to see
the CUDA compute capability of your GPU. The parameter *LLVM_TARGETS_TO_BUILD* was changed compared to the tutorial to compile only for the required platforms.


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
