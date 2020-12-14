# Introduction

This is a repo used to save tests to check the use of OpenMP with GPU offloading.
Since I had a new computing environment, 1st i checked it running CUDA examples,
then I tested OpenMP examples for the host, and finally I tested OpenMP offloading (*target* omp keyword).

# GPU monitors

Use them to 'see' the examples running.

- nsight-sys: for GPU
- stacer; for CPU

# Used compilers

- **cuda examples**: NVCC v10.2
- **openmp_host** and **openmp_gpu**: Using clang 10, available in Ubuntu 18.04;

More information about the Clang compiler setup for GPU offloading is present in the dir *openmp_gpu*.

# Dir contents

 - **cuda examples**: example used to test the cuda environment install. Gencode adapted to sm_30 (Quadro K4000). Check the makefiles for 'SMS' to replace Gencode for other GPUs.
 - **openmp_host**: examples used to test the openmp environment install. Using multithread capability;
- **openmp_gpu**: examples used to test OpenMP with GPUs (*target* keyword).

# References to build these tests

 - CUDA examples: some examples distributed with CUDA toolkit;
 - [OpenMP](https://wrf.ecse.rpi.edu/wiki/ParallelComputingSpring2014/openmp/people.sc.fsu.edu/openmp/openmp.html);
 - OpenMP GPU (target):
    - [OpenMP Application Programming Interface Examples - Version 5.0.0](https://www.openmp.org/wp-content/uploads/openmp-examples-5.0.0.pdf)
    - http://cacs.usc.edu/education/cs596/OMPtarget.pdf
    - http://cacs.usc.edu/education/cs653/OpenMP4.5_3-20-19.pdf
    - https://on-demand.gputechconf.com/gtc/2016/presentation/s6510-jeff-larkin-targeting-gpus-openmp.pdf
