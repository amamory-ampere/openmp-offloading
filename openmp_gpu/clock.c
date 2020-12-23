/*
EXAMPLE SOURCE : 
https://forums.developer.nvidia.com/t/reading-globaltimer-register-or-calling-clock-clock64-in-loop-prevent-concurrent-kernel-execution/48600/8
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock64
https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

generating Asm PTX code
https://developer.nvidia.com/blog/cuda-pro-tip-view-assembly-code-correlation-nsight-visual-studio-edition/
https://stackoverflow.com/questions/20482686/how-to-get-the-assembly-code-of-a-cuda-kernel
$ nvcc -ptx -o kernel.ptx kernel.cu

.func  (.param .b64 func_retval0) clock64(

)
{
        .reg .b64       %rd<3>;


        // inline asm
        mov.u64         %rd1, %clock64;
        // inline asm
        mov.b64         %rd2, %rd1;
        st.param.b64    [func_retval0+0], %rd2;
        ret;
}

OPENMP:
https://www.openmp.org/spec-html/5.0/openmpsu161.html#x200-9710003.4.2
https://gcc.gnu.org/onlinedocs/libgomp/omp_005fget_005fwtick.html#omp_005fget_005fwtick
omp_get_wtick

COMPILATION:
clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda  clock.c -o clock

THIS EXAMPLE WORKS, WITH SOME LITTLE EXTRA TIME
*/
#include <stdio.h>
#include <omp.h>

#define DELAY_VAL 10000000ULL // equiv to usec 

int main(void) {
  int isHost = 0;
  clock_t ck_start = clock();

  #pragma omp target map(from: isHost)
  { isHost = omp_is_initial_device(); 
    for(long long int i=0;i<DELAY_VAL;i++);
  
  }

  if (isHost < 0) {
    printf("Runtime error, isHost=%d\n", isHost);
  }

  // CHECK: Target region executed on the device
  printf ("Kernel: %ld clicks.\n", clock()-ck_start);
  printf("Target region executed on the %s\n", isHost ? "host" : "device");

  return isHost;
}
