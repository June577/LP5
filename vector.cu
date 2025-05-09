#include <iostream>
#include <cuda.h>

using namespace std;

#define N 1000000 // size of vectors

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    // Host vectors
    float *h_a, *h_b, *h_c;

    // Device vectors
    float *d_a, *d_b, *d_c;

    size_t bytes = N * sizeof(float);

    // Allocate memory on host
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Allocate memory on device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print part of the result for verification
    cout << "Sample results: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// nvcc vector_add.cu -o vector_add
// ./vector_add
