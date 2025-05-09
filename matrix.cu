% % cu
#include <iostream>
#include <cuda.h>

    using namespace std;

#define N 16 // Changeable: for demonstration, small matrix

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(int *A, int *B, int *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        int sum = 0;
        for (int k = 0; k < width; ++k)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main()
{
    int size = N * N * sizeof(int);

    // Host matrices
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1; // Example: fill with 1s
        h_B[i] = 2; // Example: fill with 2s
    }

    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result (partial)
    cout << "Resultant Matrix C (partial):\n";
    for (int i = 0; i < N && i < 5; i++)
    {
        for (int j = 0; j < N && j < 5; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
        cout << endl;
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// nvcc matrix_mul.cu -o matrix_mul
// ./matrix_mul
