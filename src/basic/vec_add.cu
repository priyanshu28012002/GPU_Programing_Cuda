#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N = 1500000; // ~1M elements
    size_t size = N * sizeof(float);

    // Host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel config (good default)
    int threadsPerBlock = 1024;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << numBlocks << " <-numBlocks  threadsPerBlock-> " << threadsPerBlock << "\n";

    vecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < 5; i++)
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Kernel Time: " << ms << " ms\n";
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}