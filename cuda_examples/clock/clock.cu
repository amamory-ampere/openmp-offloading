/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.

CLOCK

https://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel
https://wlandau.github.io/gpu/lectures/cudac-memory/cudac-memory.pdf
https://stackoverflow.com/questions/27065862/what-is-the-clock-measure-by-clock-and-clock64-in-cuda
https://forums.developer.nvidia.com/t/clock64-reversed/42709
https://forums.developer.nvidia.com/t/number-of-gpu-clock-cycles/30462/5
https://nvidia.github.io/libcudacxx/standard_api/time_library/chrono.html
https://forums.developer.nvidia.com/t/reading-globaltimer-register-or-calling-clock-clock64-in-loop-prevent-concurrent-kernel-execution/48600/8

https://stackoverflow.com/questions/31058850/overlap-kernel-execution-on-multiple-streams

 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#define DELAY_VAL 5000000000ULL

__global__ void clock_block(clock_t *d_o, const clock_t *clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < *clock_count)
    {
        clock_offset = clock() - start_clock;
    }
     d_o[0] = clock_offset;
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    clock_t *h_clock_out = (clock_t *)malloc(sizeof(clock_t));
    clock_t *h_clock_in = (clock_t *)malloc(sizeof(clock_t));
    clock_t h_clock_elapsed ;

    clock_t *d_clock_out = NULL;
    clock_t *d_clock_in = NULL;
    cudaMalloc((void **)&d_clock_out, sizeof(clock_t));
    cudaMalloc((void **)&d_clock_in, sizeof(clock_t));

    *h_clock_in = clock() + DELAY_VAL;
    cudaMemcpy(d_clock_in, h_clock_in, sizeof(clock_t), cudaMemcpyHostToDevice);


    // Launch the Vector Add CUDA Kernel
    //int threadsPerBlock = 256;
    //int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("CUDA kernel launch \n");
    clock_block<<<1, 1>>>(d_clock_out, d_clock_in);
    //clock_block<<<blocksPerGrid, threadsPerBlock>>>(d_clock_out, d_clock_in);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch clock kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_clock_out, d_clock_out, sizeof(clock_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy clock_out from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    h_clock_elapsed = clock() - *h_clock_in;
    printf ("Elapsed: %ld clicks (%f seconds).\n",h_clock_elapsed,((float)h_clock_elapsed)/CLOCKS_PER_SEC);
    h_clock_elapsed = *h_clock_out - *h_clock_in;
    printf ("Kernel: %ld clicks (%f seconds).\n", h_clock_elapsed,((float)h_clock_elapsed)/CLOCKS_PER_SEC);


    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_clock_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_clock_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_clock_in);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_clock_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free host memory
    free(h_clock_in);
    free(h_clock_out);

    printf("Done\n");
    return 0;
}

