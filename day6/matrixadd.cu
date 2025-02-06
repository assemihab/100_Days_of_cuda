#include <iostream>
#include <cuda_runtime.h>

#define N 16

__global__ void matrixAdd(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

int main() {
    int size = N * N * sizeof(int);
    int a[N][N], b[N][N], c[N][N];
    int *d_a, *d_b, *d_c;

    // Initialize matrices a and b
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices a and b to the device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(N, N);
    matrixAdd<<<1, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy the result matrix c back to the host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}