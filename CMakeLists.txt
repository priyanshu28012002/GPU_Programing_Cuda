cmake_minimum_required(VERSION 3.18)

project(CudaBasic LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: set GPU architecture (important for performance)
# For RTX 3080 → compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)



add_executable(basic_print src/basic/print.cu)
add_executable(basic_vec_add src/basic/vec_add.cu)
add_executable(basic_shared_mem src/basic/shared_mem.cu)


# Optional flags (useful for debugging)
target_compile_options(basic_print PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_compile_options(basic_vec_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_compile_options(basic_shared_mem PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Link CUDA runtime (usually automatic, but explicit is safer)
target_link_libraries(basic_print PRIVATE cuda)
target_link_libraries(basic_print PRIVATE cuda)
target_link_libraries(basic_shared_mem PRIVATE cuda)