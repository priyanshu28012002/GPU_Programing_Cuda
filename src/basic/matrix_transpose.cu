#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE 16

__global__ void matrix_transpose_kernel(
    const float* input, float* output,
    int rows, int cols)
{
    __shared__ float tile[TILE][TILE + 1]; // avoid bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load
    if (x < cols && y < rows)
    {
        tile[threadIdx.y][threadIdx.x] =
            input[y * cols + x];
    }

    __syncthreads();

    // Transposed coordinates
    int transposed_x = blockIdx.y * TILE + threadIdx.x;
    int transposed_y = blockIdx.x * TILE + threadIdx.y;

    // Store
    if (transposed_x < rows && transposed_y < cols)
    {
        output[transposed_y * rows + transposed_x] =
            tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(
        (cols + TILE - 1) / TILE,
        (rows + TILE - 1) / TILE
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, cols
    );

    cudaDeviceSynchronize();
}

int main()
{
    int rows = 4;
    int cols = 5;

    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = cols * rows * sizeof(float);

    // Host memory
    std::vector<float> h_input(rows * cols);
    std::vector<float> h_output(cols * rows);

    // Initialize input
    for (int i = 0; i < rows * cols; i++)
        h_input[i] = static_cast<float>(i);

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);

    // Call kernel
    solve(d_input, d_output, rows, cols);

    cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // Print input
    std::cout << "Input:\n";
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            std::cout << h_input[i * cols + j] << " ";
        std::cout << "\n";
    }

    // Print output
    std::cout << "\nTransposed:\n";
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
            std::cout << h_output[i * rows + j] << " ";
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}