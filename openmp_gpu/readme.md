# Introduction

This document explains how to setup OpenMP for GPU offloading in both X86-64 computer (refered as host computer) and in a NVidia Xavier board.

# Hardware requirements

In the host computer, it's necessary to check the *Computing Capability* of the existing GPU.
In the [clang/CmakeList.txt](https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/clang/CMakeLists.txt#L297) one will find 
these comments that say that the minimum hardware requirement for GPU offloading with OpenMP is 
a GPU with *Computing Capability 3.5*, referred as *sm_35*.  Please check [wikipedia](https://en.wikipedia.org/wiki/CUDA) to see the CUDA compute capability of your GPU.

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


## Dependencies

These dependencies are required for X86-64 and for Xavier.

```
$ sudo apt install build-essential
$ sudo apt install cmake
$ sudo apt install -y libelf-dev libffi-dev
$ sudo apt install -y pkg-config
$ sudo apt-get install -y ninja-build 
```

*ccmake* is not mandatory but recommended to inspect the CMake configuration.

```
$ sudo apt-get install cmake-curses-gui
```

## Clang for the Host computer

In the host computer, follow these instructions to download Clang:

```
$ git clone https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/10.x
$ mkdir build; cd build
```

There is no actual constraint regarding the Clang version for X86-64. Version 10 is selected but it could also be version 11, for example. However, for Xavier there is a constraint, as explained next.
As stated in the [libomptarget](https://github.com/llvm/llvm-project/tree/release/10.x/openmp/libomptarget), the supported compilers are **clang version 3.7 or later** and **gcc version 4.8.2 or later**.


Next, let's configure the CMAKE building system, with option for OpenMP offloading:

```
$ cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang10 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ../llvm
$ ninja
```

It is highly recommended to **use ninja build system instead of make**. The generated Makefile
has some sort of weird bug that, when you type 'make install', it compiles all over again... :/

The options *-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"* and *-DCMAKE_BUILD_TYPE=Release* are important to reduce compilation time. The parameter *LLVM_TARGETS_TO_BUILD* compile only for the required platforms. The argument *-DBUILD_SHARED_LIB=ON* is a good idea if the computer has less than 16GBytes of RAM. In a PC, the compilation time is about 35 min using **make -j 8**.

The options below are recommended for offloading debug:
- -DLIBOMPTARGET_ENABLE_DEBUG=ON 
- -DLIBOMPTARGET_NVPTX_DEBUG=ON 

There are other LLVM projects that might be of interest. So, enable them individually as in this example:

```
$ CMAKE .... -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld;openmp" ...
```

## Clang for the Xavier

Clang for Xavier (aarch64) can be version 11, without applying any modification. 
On the other hand, if for any reason an older LLVM version is required, the modification is simple.

If one tries to compile LLVM version 10 or earlier for aarch64, the following message will appear hidden among tons of other messages:

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

Next, it is the LLVM/Clang configuration itself. For the [NVidia Xaxier](https://releases.llvm.org/11.0.0/docs/HowToBuildOnARM.html) board, the configuration command is: 

```
cmake -G Ninja -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_72 -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=72 -DLIBOMPTARGET_ENABLE_DEBUG=ON -DLIBOMPTARGET_NVPTX_DEBUG=ON -DLLVM_ENABLE_PROJECTS="clang;openmp" -DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/clang11 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a" ../llvm
```

-DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_72 -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=72

Where the main difference is the aarch64 and NVIDIA build targets *-DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX"* and the ARM CPU architecture *-DCMAKE_C_FLAGS="-march=armv8.2-a" -DCMAKE_CXX_FLAGS="-march=armv8.2-a"* according to the Xavier spec.

Before running the actual compilation in Xavier, it is recommended to turn on the performance
mode. This way the 8 cores will be available to be used for the compilation, otherwise, only 4 cores are available for default. The 1st command displays the current power mode and the second allows to change the power mode. 

```
$ sudo /usr/sbin/nvpmodel -q
$ sudo /usr/sbin/nvpmodel -m <x>
```

Where <x> is the power mode ID (i.e. 0, 1, 2, 3, 4, 5, or 6).

Finally, run the compilation:

```
$ nohup nice -5 ninja
$ ninja install
```

The command *nohup* is recommend in case the terminal to Xavier is closed. *nice -5* is recommend to reduce the priority of the compilation since Ninja uses all available cores for default.

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


The second test is with an actual OpenMP application with *target* keyword.
There is a minimal offloading example at https://freecompilercamp.org/llvm-openmp-build/.
Then, the application is profiled with *nvprof* to detect the actual use of the GPU.
Assuming a source code called *omp-test.c*, the compilation program for GPU offloading is:

```
$ clang  -o omp-test omp-test.c -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
```

When compiling the application, you will get a warning like this:

```
clang-11: warning: No library 'libomptarget-nvptx-sm_72.bc' found in the default clang lib directory or in LIBRARY_PATH. Expect degraded performance due to no inlining of runtime functions on target devices. [-Wopenmp-target]
```

According to [this source](https://hpc-wiki.info/hpc/Building_LLVM/Clang_with_OpenMP_Offloading_to_NVIDIA_GPUs), this will reduce the GPU performance. 
However, **I could not make OpenMP offloading work with this warning**. 
Luckily, this can be easily fixed by recompiling again OpenMP using Clang as compiler instead of GCC.
You repeat the same CMake configuration command used before and just replace the arguments related to the compiler, like this:

```
	-DCMAKE_C_COMPILER=$(CLANG_PATH)/bin/clang \
	-DCMAKE_CXX_COMPILER=$(CLANG_PATH)/bin/clang++ \
```

For convenience, it is suggested to edit the user's and root's *.bashrc* with the following definitions:


```
# setting up Clang 11 with GPU offloading capability
export PATH=/opt/clang11/bin:$PATH
export LD_LIBRARY_PATH=/opt/clang11/lib:$LD_LIBRARY_PATH

# setting up CUDA stuff
export PATH=/usr/local/cuda-10.2/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64/:$LD_LIBRARY_PATH
```

It's also suggested to alter this file */etc/skel/.bashrc*.

## Checking Clang and libomptarget

There are several types of checks to be applied. Here is a description from the most basic to the most complete tests. 

When running *ldd*, it's possible to see that *libomptarget.so* and *libomp.so* were correctly linked.

```
$ ldd ./omp-test
	linux-vdso.so.1 (0x0000007f8f990000)
	libomp.so => /home/alexandre/tools/clang11/lib/libomp.so (0x0000007f8f88f000)
	libomptarget.so => /home/alexandre/tools/clang11/lib/libomptarget.so (0x0000007f8f86e000)
	libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007f8f811000)
	libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007f8f6b8000)
	libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007f8f6a3000)
	/lib/ld-linux-aarch64.so.1 (0x0000007f8f964000)
	libstdc++.so.6 => /usr/lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000007f8f50f000)
	libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007f8f4eb000)
	libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007f8f432000)
```


Next, execute the profiler to check for actual GPU use.

```
$ nvprof --print-gpu-trace ./omp-test
==1532== NVPROF is profiling process 1532, command: ./omp-test
==1532== Warning: ERR_NVGPUCTRPERM - The user does not have permission to profile on the target device. See the following link for instructions to enable permissions and get more information: https://developer.nvidia.com/ERR_NVGPUCTRPERM 
==1532== Profiling application: ./omp-test
==1532== Profiling result:
No kernels were profiled.
```

When it does not work, the resulting message is like this:

```
$ nvprof --print-gpu-trace ./omp-ser-cl
======== Warning: No CUDA application was profiled, exiting
```

In the first message CUDA was detected, although the user has no privileges to execute *nvprof*.While in the second message no CUDA application was detected.

This permission warning for GPU profiling can be easily solved with the following [commands](- https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag). However, for Jetson boards the instructions are a little bit different as [discussed here](https://forums.developer.nvidia.com/t/nvprof-needs-to-be-run-as-root-user-for-profiling-on-jetson/115293).

The required command to give profiling permission for users is:

```
$ modprobe nvgpu NVreg_RestrictProfilingToAdminUsers=0
```

[Jetson Stats](https://github.com/rbonghi/jetson_stats) is another application to monitor the Jetson GPU that could be used to check if the GPU offloading is really happening.

Please not that executing the OpenMp application is not enough to check if the offloading is working. By default, *libomp* has a fall back mechanism that, if the GPU offloading fails, then
it will execute the app as a multithread app. This way, you wont really see any difference if the GPU is being used or not. You need these extra tools (nvprof, Jetson Stats, etc) to check if the GPU is being used. However, it is possible also to disable this fallback mechanism by setting the environment variable [OMP_TARGET_OFFLOAD](https://www.openmp.org/spec-html/5.0/openmpse65.html).

If *OMP_TARGET_OFFLOAD* is set and offloading fails, you should see something like this:

```
$ export OMP_TARGET_OFFLOAD=MANDATORY
$ ./omp-test
Libomptarget fatal error 1: failure of target construct while offloading is mandatory
```

If you have sudo or the permissions to run profile, you will see this output after running the OpenMP app:

```
$ sudo su
$ nvprof ./omp-test
==19911== NVPROF is profiling process 19911, command: ./omp-test
==19911== Warning: Unified Memory Profiling is not supported on the underlying platform. System requirements for unified memory can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
==19911== Profiling application: ./omp-test
==19911== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   95.05%  65.471ms         1  65.471ms  65.471ms  65.471ms  __omp_offloading_b301_163dae_main_l35
                    3.47%  2.3900ms         2  1.1950ms  4.1940us  2.3858ms  [CUDA memcpy DtoH]
                    1.48%  1.0173ms         3  339.10us  1.8890us  523.58us  [CUDA memcpy HtoD]
      API calls:   53.09%  233.30ms         1  233.30ms  233.30ms  233.30ms  cuDevicePrimaryCtxRetain
                   17.06%  74.983ms         1  74.983ms  74.983ms  74.983ms  cuModuleLoadDataEx
                   15.01%  65.976ms         4  16.494ms  16.704us  65.535ms  cuStreamSynchronize
                    9.18%  40.354ms         1  40.354ms  40.354ms  40.354ms  cuDevicePrimaryCtxRelease
                    4.13%  18.134ms         1  18.134ms  18.134ms  18.134ms  cuModuleUnload
...
```

These lines show the execution time of each CUDA interface function implemented below OpenMP.

The next step in terms of testing is to perform the tests of the LLVM distribution.
However, these tests require Clang version 6 or earlier. To run those tests, execute:


```
$ ninja check-libomptarget

...
********************
********************
Failed Tests (3):
  libomptarget :: api/omp_get_num_devices_with_empty_target.c
  libomptarget :: mapping/declare_mapper_api.cpp
  libomptarget :: mapping/pr38704.c


Testing Time: 44.09s
  Passed: 19
  Failed:  3
```

```
$ ninja check-libomptarget-nvptx 

...

********************
********************
Failed Tests (5):
  libomptarget-nvptx :: api/get_max_threads.c
  libomptarget-nvptx :: data_sharing/alignment.c
  libomptarget-nvptx :: parallel/level.c
  libomptarget-nvptx :: parallel/nested.c
  libomptarget-nvptx :: parallel/num_threads.c


Testing Time: 69.49s
  Passed: 8
  Failed: 5
```

Few tests failed. Addition investigation is required to check the cause of these failed tests.

Other relevant tests include:

```
$ ninja check-openmp 
$ ninja check-libomp
```

## Debugging libomptarget


The define *LIBOMPTARGET_ENABLE_DEBUG* enables debug messages for libomptarget. Add this definition in CMake as in this example to recompile libomptarget. 

LIBOMPTARGET_NVPTX_DEBUG

```
$ cmake ... -DLIBOMPTARGET_ENABLE_DEBUG=ON -DLIBOMPTARGET_NVPTX_DEBUG=ON ...
```

As stated in the [readme](https://github.com/llvm/llvm-project/tree/release/10.x/openmp/libomptarget), it is possible to compile only libomptarget.



--> You need to compile libomp with -DOMPTARGET_DEBUG so that debug output is enabled.
https://github.com/clang-ykt/clang/issues/14#issuecomment-301114816


After all testing and debugging is done, it's suggested to recompile Clang removing the libomptarget's debug attributes. An alternative is to keep two Clang installed. 
One for debugging and another for performance evaluations.

```
-DLIBOMPTARGET_ENABLE_DEBUG=OFF -DLIBOMPTARGET_NVPTX_DEBUG=OFF
```


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


# Compiling the OpenMp application

Make sure that both CUDA and Clang are with their required environment variables (PATH and LD_LIBRARY_PATH) set as explained during the testing phase


Then, compile the OpenMP code with *target*:

```
$ cd /tmp
$ clang -v -o <exec> source_code.c -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
```

Then, type this command to check if the dynamic libraries where found.

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


# Profiling

In the 2020.3 release, Nsight Systems adds ability to [analyze applications parallelized using OpenMP](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#openmp-trace).

[nsight-systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-linux)
[nsys CLI commands](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling)

## Remote profiling 

[Connecting to the Target Device](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#linux-connect-target)


https://docs.nvidia.com/cuda/profiler-users-guide/index.html#remote-profiling
NVIDIA Tegra System Profiler - Introduction - 2015
https://youtu.be/R_GzhZe8IcM?t=32
https://www.youtube.com/watch?v=bda5Er27jqo&list=RDCMUCBHcMCGaiJhv-ESTcWGJPcw&index=19


# Extending 

[NVIDIA Tools Extension (NVTX)](https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm)


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
