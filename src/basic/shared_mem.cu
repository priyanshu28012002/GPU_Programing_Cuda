#include <iostream>
#include <cuda_runtime.h>

__global__ void copyKernel(float* g_data, int N)
{
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N)
    {
        // Global → Shared
        s_data[tid] = g_data[gid];

        __syncthreads();

        // Modify data in shared memory
        s_data[tid] *= 2.0f;

        __syncthreads();

        // Shared → Global
        g_data[gid] = s_data[tid];
    }
}

int main()
{
    int N =  1024; // 1024 elements
    size_t size = N * sizeof(float);

    // Host memory
    float* h_data = new float[N];

    // Initialize data
    for (int i = 0; i < N; i++)
        h_data[i] = i * 1.0f;

    // Device memory
    float* d_data;
    cudaMalloc(&d_data, size);

    // Copy to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Kernel config
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    copyKernel<<<numBlocks, threadsPerBlock>>>(d_data, N);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "First 10 results:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << "Index " << i << " : " << h_data[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}