# LP5_HPC

openmp:
## gedit reduction.cpp
g++ reduction.cpp -o reduction -fopenmp
./reduction

Cuda:
gedit cuda2.cu
nvcc cuda2.cu -o cuda2
./cuda2
