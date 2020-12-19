clang -o mxm   mxm.c         -fopenmp=libomp 
clang -o helm  helmholtz.c   -fopenmp=libomp -lm
clang -o dot   dot_product.c -fopenmp=libomp -lm

echo "Compilation FINISHED !"


sudo nvprof ./mxm
sudo nvprof ./helm
sudo nvprof ./dot

echo "You may run the tests again with different number of threads with:"
echo "\$ export OMP_NUM_THREADS=5"
