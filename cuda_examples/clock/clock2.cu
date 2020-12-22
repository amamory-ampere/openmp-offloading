/*
EXAMPLE SOURCE : 
https://forums.developer.nvidia.com/t/reading-globaltimer-register-or-calling-clock-clock64-in-loop-prevent-concurrent-kernel-execution/48600/8

COMPILATION:
/usr/local/cuda-10.2/bin/nvcc -ccbin g++ -I../common/inc  -m64 -g -G    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -o clock2.o -c clock2.cu
/usr/local/cuda-10.2/bin/nvcc -ccbin g++   -m64 -g -G      -gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -o clock2 clock2.o

THIS EXAMPLE WORKS, WITH SOME LITTLE EXTRA TIME
*/
#include <stdio.h>

#define DELAY_VAL 5000000000ULL // about 5 secs

long milliseconds()
{
    long            ms; // Milliseconds
    time_t          s;  // Seconds
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
    return ms + s *1000;
}

__global__ void child(){

    unsigned long long start = clock64();
    while (clock64()< start+DELAY_VAL);
}

int main(int argc, char* argv[]){

    cudaStream_t st1, st2;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);

    clock_t ck_start = clock();
    long start = milliseconds();
    long now = 0;
    child<<<1,1,0,st1>>>();
    /*
    if (argc > 1){
        printf("running double kernel\n");
        while ( now < start + DELAY_VAL) {
            now = milliseconds();
        }
        //parent<<<1,1,0,st2>>>();
    }*/
    cudaDeviceSynchronize();
    printf ("Elapsed: %ld clicks.\n",milliseconds()-start);
    printf ("Kernel: %ld clicks (%f seconds).\n", clock()-ck_start,((float)clock()-ck_start)/CLOCKS_PER_SEC);
    return now;
}