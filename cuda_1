#include<math.h>
#include<time.h>
#include<iostream>
#include "cuda_runtime.h"

void cpuSum(int* A, int*B, int* C, int N)
    {
    for (int i=0; i<N; ++i)
        {
        C[i] = A[i] + B[i];
        }
    }
    
__global__ void kernel(int* A, int* B, int* C, int N)
    {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N)
        {
        C[i] = A[i] + B[i];
        }
    }
    
void gpuSum(int* A, int* B, int* C, int N)
    {
    int threadsPerBlock = min(1024, N);
    int blocksPerGrid = ceil(double(N)/double(threadsPerBlock));
    kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    }
    
bool isVectorEqual(int* A, int* B, int N)
    {
    for (int i=0; i<N; ++i)
        {
        if (A[i] != B[i])
            {
            return false;
            }
        }
    return true;
    }
    
int main()
    {
    int N = 2e7;
    int *A, *B, *C, *D, *a, *b, *c;
    int size = N * sizeof(int);
    
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    D = (int*)malloc(size);
    
    for (int i=0; i<N; ++i)
        {
        A[i] = rand()%1000;
        B[i] = rand()%1000;
        }
        
    clock_t start, end;
    
    start = clock();
    cpuSum(A, B, C, N);
    end = clock();
    float timeTakenCPU = ((float)(end-start))/CLOCKS_PER_SEC;
    
    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);
    
    cudaMemcpy(a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, size, cudaMemcpyHostToDevice);
    
    start = clock();
    gpuSum(a, b, c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(D, c, size, cudaMemcpyDeviceToHost);
    end = clock();
    float timeTakenGPU = ((float)(end-start))/CLOCKS_PER_SEC;
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    bool success = isVectorEqual(C, D, N);
    
    printf("Vector Addition\n");
    printf("--------------------\n");
    printf("CPU Time: %f \n", timeTakenCPU);
    printf("GPU Time: %f \n", timeTakenGPU);
    printf("Speed Up: %f \n", timeTakenCPU/timeTakenGPU);
    printf("Verification: %s \n", success ? "true":"false");
    }
