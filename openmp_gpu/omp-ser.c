/*
compilation:
clang -pedantic -Wall -o omp-ser-cl omp-ser.c -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_30 -lc -lcudart -L/usr/local/cuda/lib64 --cuda-path=/usr/local/cuda

source: based on "OpenMP Application Programming Interface Examples - Version 5.0.0 - November 2019"
Section 4.7.1 'Simple target data Construct'
*/
#include <stdlib.h>
#include <omp.h>

int main (){
    const int N = 1000000;
    unsigned int i;
    unsigned int j;
    float *a;
    float *b;
    float *c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));
        //srand((unsigned int)time(NULL));

    for (i = 0; i < N; i++) {
        a[i] = 0;
        b[i] = ((float)rand() / (float)(RAND_MAX)) * 4.0;
        c[i] = ((float)rand() / (float)(RAND_MAX)) * 4.0;
    }

    #pragma omp target data map (to: c[0:N], b[0:N]) map(from: a[0:N])
    {
        #pragma omp target
        #pragma omp parallel for
        for (j=0; j<N; j++)
        {
            a[j] = b[j]+3.73*c[j];
        }
    }

    return 0;
}
