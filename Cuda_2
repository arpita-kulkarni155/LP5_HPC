#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

void cpuMatrixMultiply(int* A, int* B, int* C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

__global__ void gpuMatrixMultiply(int* A, int* B, int* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int value = 0;
        for (int k = 0; k < N; ++k)
        {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void gpuMatrixMultiplyLauncher(int* A, int* B, int* C, int N)
{
    int threadsPerBlock = 16;
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks(ceil(float(N) / threadsPerBlock), ceil(float(N) / threadsPerBlock));

    gpuMatrixMultiply<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

bool isMatrixEqual(int* A, int* B, int N)
{
    for (int i = 0; i < N * N; ++i)
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
    int N = 512;
    int *A, *B, *C, *D, *a, *b, *c;
    int size = N * N * sizeof(int);

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    D = (int*)malloc(size);

    for (int i = 0; i < N * N; ++i)
    {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    clock_t start, end;

    start = clock();
    cpuMatrixMultiply(A, B, C, N);
    end = clock();
    float timeTakenCPU = ((float)(end - start)) / CLOCKS_PER_SEC;

    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);

    cudaMemcpy(a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, size, cudaMemcpyHostToDevice);

    start = clock();
    gpuMatrixMultiplyLauncher(a, b, c, N);
    cudaMemcpy(D, c, size, cudaMemcpyDeviceToHost);
    end = clock();
    float timeTakenGPU = ((float)(end - start)) / CLOCKS_PER_SEC;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    bool success = isMatrixEqual(C, D, N);

    printf("Matrix Multiplication\n");
    printf("--------------------\n");
    printf("CPU Time: %f seconds\n", timeTakenCPU);
    printf("GPU Time: %f seconds\n", timeTakenGPU);
    printf("Speed Up: %f\n", timeTakenCPU / timeTakenGPU);
    printf("Verification: Matrices are %s\n", success ? "equal" : "not equal");

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}

