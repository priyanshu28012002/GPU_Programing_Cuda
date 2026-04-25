cmake_minimum_required(VERSION 3.18)

project(CudaBasic LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: set GPU architecture (important for performance)
# For RTX 3080 → compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenCV REQUIRED)


set(CUDA_PROGRAMS
    basic_print:src/basic/print.cu
    basic_vec_add:src/basic/vec_add.cu
    basic_shared_mem:src/basic/shared_mem.cu
    basic_matrixMultiplication:src/basic/matrixMultiplication.cu
    basic_matrix_transpose:src/basic/matrix_transpose.cu
    inverseImage:src/inverseImage/main.cu


)

foreach(entry ${CUDA_PROGRAMS})
    string(REPLACE ":" ";" parts ${entry})
    list(GET parts 0 target)
    list(GET parts 1 source)

    add_executable(${target} ${source})

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    target_link_libraries(${target} PRIVATE cuda)
endforeach()
# Include directories
target_include_directories(inverseImage PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(inverseImage PRIVATE ${OpenCV_LIBS})