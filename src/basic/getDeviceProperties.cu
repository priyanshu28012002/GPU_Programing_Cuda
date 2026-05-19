// nvcc getDeviceProperties.cu -o info


#include <iostream>
#include <cuda_runtime.h>



void printDeviceProperties(const cudaDeviceProp& p) {
    std::cout << "===== GPU DEVICE INFO =====\n";

    std::cout << "Name: " << p.name << "\n";
    std::cout << "Compute Capability: " << p.major << "." << p.minor << "\n";

    std::cout << "\n--- Memory ---\n";
    std::cout << "Total Global Memory (GB): " << p.totalGlobalMem / (1024.0*1024*1024) << "\n";
    std::cout << "Shared Memory per Block: " << p.sharedMemPerBlock << "\n";
    std::cout << "Shared Memory per Multiprocessor: " << p.sharedMemPerMultiprocessor << "\n";
    std::cout << "Total Constant Memory: " << p.totalConstMem << "\n";
    std::cout << "L2 Cache Size: " << p.l2CacheSize << "\n";
    std::cout << "Memory Clock Rate (kHz): " << p.memoryClockRate << "\n";
    std::cout << "Memory Bus Width (bits): " << p.memoryBusWidth << "\n";

    std::cout << "\n--- Execution ---\n";
    std::cout << "Multiprocessor Count (SMs): " << p.multiProcessorCount << "\n";
    std::cout << "Max Threads per Block: " << p.maxThreadsPerBlock << "\n";
    std::cout << "Max Threads per Multiprocessor: " << p.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Warp Size: " << p.warpSize << "\n";
    std::cout << "Max Blocks per SM (theoretical limit depends): N/A (dynamic)\n";

    std::cout << "\n--- Dimensions ---\n";
    std::cout << "Max Threads Dim: "
              << p.maxThreadsDim[0] << ", "
              << p.maxThreadsDim[1] << ", "
              << p.maxThreadsDim[2] << "\n";

    std::cout << "Max Grid Size: "
              << p.maxGridSize[0] << ", "
              << p.maxGridSize[1] << ", "
              << p.maxGridSize[2] << "\n";

    std::cout << "\n--- Registers ---\n";
    std::cout << "Registers per Block: " << p.regsPerBlock << "\n";

    std::cout << "\n--- Clock ---\n";
    std::cout << "Core Clock Rate (kHz): " << p.clockRate << "\n";

    std::cout << "\n--- Texture ---\n";
    std::cout << "Texture Alignment: " << p.textureAlignment << "\n";

    std::cout << "\n--- PCI / System ---\n";
    std::cout << "PCI Bus ID: " << p.pciBusID << "\n";
    std::cout << "PCI Device ID: " << p.pciDeviceID << "\n";

    std::cout << "\n--- Features ---\n";
    std::cout << "Unified Addressing: " << p.unifiedAddressing << "\n";
    std::cout << "Managed Memory: " << p.managedMemory << "\n";
    std::cout << "Concurrent Kernels: " << p.concurrentKernels << "\n";
    std::cout << "ECC Enabled: " << p.ECCEnabled << "\n";
    std::cout << "Integrated GPU: " << p.integrated << "\n";
    std::cout << "Can Map Host Memory: " << p.canMapHostMemory << "\n";
    std::cout << "Compute Mode: " << p.computeMode << "\n";

    std::cout << "\n--- Async Engine ---\n";
    std::cout << "Async Engine Count: " << p.asyncEngineCount << "\n";

    std::cout << "\n============================\n";
}

int main() {
    int count;
    cudaGetDeviceCount(&count);

    std::cout << "Total CUDA Devices: " << count << "\n\n";

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "DEVICE " << i << ":\n";
        printDeviceProperties(prop);
    }

    return 0;
}