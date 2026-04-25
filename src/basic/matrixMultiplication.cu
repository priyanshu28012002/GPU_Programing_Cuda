#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#define TILE 32

// ================= KERNEL =================
__global__ void matrix_multiplication_kernel(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t)
    {
        int A_col = t * TILE + threadIdx.x;
        int B_row = t * TILE + threadIdx.y;

        // Load A tile
        if (row < M && A_col < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (B_row < N && col < K)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

// Compute
#pragma unroll
        for (int i = 0; i < TILE; ++i)
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K)
    {
        C[row * K + col] = sum;
    }
}

// ================= SOLVE =================
extern "C" void solve(const float *d_A, const float *d_B, float *d_C,
                      int M, int N, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (K + 15) / 16,
        (M + 15) / 16);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();
}

// ================= CPU CHECK =================
void cpu_matmul(const std::vector<float> &A,
                const std::vector<float> &B,
                std::vector<float> &C,
                int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
        {
            float sum = 0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];

            C[i * K + j] = sum;
        }
}
void print_matrix(const std::vector<float> &mat, int rows, int cols, const char *name,
                  int max_rows = 8, int max_cols = 8)
{
    std::cout << "\nMatrix " << name << " (" << rows << "x" << cols << ")\n";

    int r = std::min(rows, max_rows);
    int c = std::min(cols, max_cols);

    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }

    if (rows > max_rows || cols > max_cols)
        std::cout << "... (truncated)\n";
}

// ================= MAIN =================
int main()
{
    // Change sizes for testing
    int M = 4;
    int N = 4;
    int K = 4;

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Host memory
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K, 0);
    std::vector<float> h_C_ref(M * K, 0);

    // Initialize
    for (int i = 0; i < M * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < N * K; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // Run GPU
    solve(d_A, d_B, d_C, M, N, K);

    // Copy back
    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // CPU reference
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    // Validate
    // double max_error = 0.0;
    // for (int i = 0; i < M * K; ++i)
    // {
    //     double err = std::abs(h_C[i] - h_C_ref[i]);
    //     if (err > max_error) max_error = err;
    // }

    // std::cout << "Max error: " << max_error << std::endl;

    print_matrix(h_A, M, N, "A");
    print_matrix(h_B, N, K, "B");
    print_matrix(h_C, M, K, "C (GPU)");
    print_matrix(h_C_ref, M, K, "C (CPU)");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}