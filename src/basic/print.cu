#include <iostream>
#include <cuda_runtime.h>

__global__ void debugKernel()
{
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Block %d, Thread %d, Global ID %d\n",
           blockIdx.x,
           threadIdx.x,
           global_id);
}

int main()
{
    int numBlocks = 10;
    int threadsPerBlock = 1024;

    debugKernel<<<numBlocks, threadsPerBlock>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    return 0;
}