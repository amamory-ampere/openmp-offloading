

# Host side 

First, you have to have an updated compiler with support to OpenMP. 
I am using Clang 11 from a docker image. 

Assuming you have docker installed, execute the following command to download the docker image:

```
$ docker pull silkeh/clang:11
```

Then, assuming the  directory $PWD/clang_tests is where are the sources, this command maps the sources to the dir /tmp in the docker and also map the CUDA environment from /usr/local/cuda in the host side to the same directore in the docker side. 

```
$ docker run -it -v $PWD/clang_tests:/tmp -v /usr/local/cuda:/usr/local/cuda  silkeh/clang:11 /bin/bash
```

# Docker side

In the docker terminal, type these commands:

```
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib
$ export PATH=/usr/local/cuda/bin:$PATH
$ export CUDA_HOME=/usr/local/cuda          
```

Then, compile the OpenMP code with target:

```
$ cd /tmp
$ clang -pedantic -Wall -o omp-ser-cl omp-ser.c -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -lc -lcudart -L/usr/local/cuda/lib64 --cuda-path=/usr/local/cuda
clang-11: warning: Unknown CUDA version 10.2. Assuming the latest supported version 10.1 [-Wunknown-cuda-version]
```

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

In the host, in the directory *$PWD/clang_tests*, execute:

```
$ ./omp-ser-cl
```

It is better to check if the GPU is working using [nsight-sys](https://developer.nvidia.com/nsight-systems):

```
$ /usr/local/cuda/nsight-sys
```

Then run the executable from nsight-sys to see the created threads and the GPU access.

