#include <iostream>
#include <cuda_runtime.h>

#define N 16

__global__ void matrixMulKernel(int* d_A, int* d_B, int* d_C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; ++k) {
            sum += d_A[row * width + k] * d_B[k * width + col];
        }
        d_C[row * width + col] = sum;
    }
}

void matrixMul(int* h_A, int* h_B, int* h_C, int width) {
    int size = width * width * sizeof(int);
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int h_A[N * N], h_B[N * N], h_C[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() % 1024;
        h_B[i] = rand() % 1024;
    }

    matrixMul(h_A, h_B, h_C, N);

    std::cout << "Result matrix: " << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}