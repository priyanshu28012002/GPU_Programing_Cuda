// #include <iostream>

// __global__ void vecAdd(float *a, float *b, float *c, int n) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < n) c[i] = a[i] + b[i];
// }

// int main() {
//     int n = 1000;
//     size_t size = n * sizeof(float);

//     float *a, *b, *c;
//     cudaMallocManaged(&a, size);
//     cudaMallocManaged(&b, size);
//     cudaMallocManaged(&c, size);

//     for (int i = 0; i < n; i++) {
//         a[i] = i;
//         b[i] = i;
//     }

//     vecAdd<<<(n+255)/256, 256>>>(a, b, c, n);
//     cudaDeviceSynchronize();

//     std::cout << c[10] << std::endl;

//     cudaFree(a); cudaFree(b); cudaFree(c);
// }

#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "Shared Mem per Block: " << prop.sharedMemPerBlock << "\n";
    std::cout << "Registers per Block: " << prop.regsPerBlock << "\n";
    std::cout << "Memory (GB): " << prop.totalGlobalMem / (1024.0*1024*1024) << "\n";
    std::cout << "Memory Clock (kHz): " << prop.memoryClockRate << "\n";
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";

    return 0;
}