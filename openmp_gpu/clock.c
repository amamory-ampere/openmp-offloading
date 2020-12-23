/*
EXAMPLE SOURCE : 
https://forums.developer.nvidia.com/t/reading-globaltimer-register-or-calling-clock-clock64-in-loop-prevent-concurrent-kernel-execution/48600/8

https://www.openmp.org/spec-html/5.0/openmpsu161.html#x200-9710003.4.2
https://gcc.gnu.org/onlinedocs/libgomp/omp_005fget_005fwtick.html#omp_005fget_005fwtick
omp_get_wtick

COMPILATION:
gcc clock.c -o clock -fopenmp

THIS EXAMPLE WORKS, WITH SOME LITTLE EXTRA TIME
*/
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define DELAY_VAL 5000000000ULL // about 5 secs

long milliseconds()
{
    long            ms; // Milliseconds
    time_t          s;  // Seconds
    struct timespec spec;

    clock_gettime(CLOCK_MONOTONIC, &spec);

    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
    return ms + s *1000;
}

__global__ void child(){

    unsigned long long start = clock64();
    while (clock64()< start+DELAY_VAL);
}

int main(int argc, char* argv[]){

    clock_t ck_start = clock();
    long start = milliseconds();
    long now = 0;
    //child<<<1,1,0,st1>>>();
    /*
    if (argc > 1){
        printf("running double kernel\n");
        while ( now < start + DELAY_VAL) {
            now = milliseconds();
        }
        //parent<<<1,1,0,st2>>>();
    }*/
    //cudaDeviceSynchronize();
    printf ("Elapsed: %ld clicks.\n",milliseconds()-start);
    printf ("Kernel: %ld clicks (%f seconds).\n", clock()-ck_start,((float)clock()-ck_start)/CLOCKS_PER_SEC);
    return now;
}