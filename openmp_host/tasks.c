/*
 testing omp constructs to represents tasks

 https://www.openmp.org//wp-content/uploads/sc13.tasking.ruud.pdf

 compile:
 $ clang tasks.c -fopenmp=libomp -o tasks
*/
#include <stdio.h>
#include <unistd.h>

void task1(){
    int i=0;
    while(1){
        printf("task1 - %d\n", i);
        i++;
        sleep(5);
    }
}

void task2(){
    int i=0;
    while(1){
        printf("task2 - %d\n", i);
        i++;
        sleep(4);
    }
}

int main(int argc, char *argv[]) 
{ 
    #pragma omp parallel   
    {     
        #pragma omp single      
        { 
            printf("A ");         
            #pragma omp task 
            {
                //printf("car ");
                task2();
            }         
            #pragma omp task 
            {
                //printf("race ");
                task1();
            } 
            #pragma omp taskwait
            printf("is fun to watch ");
        }
    } // End of parallel region 
    printf("\n");
    return(0); 
} 