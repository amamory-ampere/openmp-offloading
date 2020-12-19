/*

source: https://wrf.ecse.rpi.edu/wiki/ParallelComputingSpring2014/openmp/people.sc.fsu.edu/openmp/openmp.html

compilation:
clang -x c++ -pedantic -Wall -o dot  dot_product.c -fopenmp=libomp -lc -lm -lstdc++

it is running in the CPU with multithreads, but not in the GPU

using clang-10
export LD_LIBRARY_PATH=/usr/lib/llvm-10/lib/:$LD_LIBRARY_PATH

#include /usr/lib/llvm-10/include/openmp/

running:
.dot

result:
$ ./dot

DOT_PRODUCT
  C/OpenMP version

  A program which computes a vector dot product.

  Number of processors available = 12
  Number of threads =              12

  Sequential avg    0.0000055289
  Parallel   avg    0.0001668859
  Speedup (%)     -96.6870008715

  Sequential avg    0.0000672054
  Parallel   avg    0.0003444409
  Speedup (%)     -80.4885477161

  Sequential avg    0.0003375340
  Parallel   avg    0.0003624392
  Speedup (%)      -6.8715546843

  Sequential avg    0.0029597402
  Parallel   avg    0.0010634923
  Speedup (%)     178.3038642697

  Sequential avg    0.0260726070
  Parallel   avg    0.0068952155
  Speedup (%)     278.1260643354

DOT_PRODUCT
  Normal end of execution.

*/

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

# include <omp.h>

int main ( int argc, char *argv[] );
double test01 ( int n, double x[], double y[] );
double test02 ( int n, double x[], double y[] );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for DOT_PRODUCT.

  Discussion:

    This program illustrates how a vector dot product could be set up
    in a C program using OpenMP.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 April 2009

  Author:

    John Burkardt
*/
{
  double factor;
  int i,j;
  int n;
  int repeat = 100;
  double avg_time_p = 0;
  double avg_time_s = 0;
  double wtime;
  double *x;
  double xdoty;
  double *y;

  printf ( "\n" );
  printf ( "DOT_PRODUCT\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "\n" );
  printf ( "  A program which computes a vector dot product.\n" );

  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
/*
  Set up the vector data.
  N may be increased to get better timing data.

  The value FACTOR is chosen so that the correct value of the dot product 
  of X and Y is N.
*/
  n = 100;


  while ( n < 10000000 )
  {
    n = n * 10;

    x = ( double * ) malloc ( n * sizeof ( double ) );
    y = ( double * ) malloc ( n * sizeof ( double ) );

    factor = ( double ) ( n );
    factor = 1.0 / sqrt ( 2.0 * factor * factor + 3 * factor + 1.0 );

    for ( i = 0; i < n; i++ )
    {
      x[i] = ( i + 1 ) * factor;
    }

    for ( i = 0; i < n; i++ )
    {
      y[i] = ( i + 1 ) * 6 * factor;
    }

    printf ( "\n" );
/*
  Test #1
*/
  avg_time_s = 0;
  for(j=0;j<repeat;j++){
    wtime = omp_get_wtime ( );

    xdoty = test01 ( n, x, y );

    wtime = omp_get_wtime ( ) - wtime;
    avg_time_s += wtime;

    //printf ( "  Sequential  %8d  %14.6e  %15.10f\n", n, xdoty, wtime );
  }
  printf ( "  Sequential avg %15.10f\n", avg_time_s/((double) repeat));
/*
  Test #2
*/
  avg_time_p = 0;
  for(j=0;j<repeat;j++){
    wtime = omp_get_wtime ( );

    xdoty = test02 ( n, x, y );

    wtime = omp_get_wtime ( ) - wtime;
    avg_time_p += wtime;

    //printf ( "  Parallel    %8d  %14.6e  %15.10f\n", n, xdoty, wtime );
  }
  printf ( "  Parallel   avg %15.10f\n", avg_time_p/((double) repeat));
  printf ( "  Speedup (\%)    %15.10f\n", (avg_time_s/avg_time_p*100.0)-100.0);
  
    free ( x );
    free ( y );
  }
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "DOT_PRODUCT\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/

double test01 ( int n, double x[], double y[] )

/******************************************************************************/
/*
  Purpose:

    TEST01 computes the dot product with no parallel processing directives.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    05 April 2008

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the vectors.

    Input, double X[N], Y[N], the vectors.

    Output, double TEST01, the dot product of X and Y.
*/
{
  int i;
  double xdoty;

  xdoty = 0.0;

  for ( i = 0; i < n; i++ )
  {
    xdoty = xdoty + x[i] * y[i];
  }

  return xdoty;
}
/******************************************************************************/

double test02 ( int n, double x[], double y[] )

/******************************************************************************/
/*
  Purpose:

    TEST02 computes the dot product with parallel processing directives.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    05 April 2008

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the vectors.

    Input, double X[N], Y[N], the vectors.

    Output, double TEST02, the dot product of X and Y.
*/
{
  int i;
  double xdoty;

  xdoty = 0.0;

# pragma omp parallel \
  shared ( n, x, y ) \
  private ( i )

# pragma omp for reduction ( + : xdoty )

  for ( i = 0; i < n; i++ )
  {
    xdoty = xdoty + x[i] * y[i];
  }

  return xdoty;
}

