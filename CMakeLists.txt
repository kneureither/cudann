cmake_minimum_required(VERSION 3.18)  # Required for CUDA support
project(cudann LANGUAGES CXX CUDA)

# Set C++ standard
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES "70") #2080Ti: 75 #A100 70 | 60 61 70 75 89

# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g -O0 ) # s -fno-omit-frame-pointer

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # Debug flags for CUDA

# Include directories for headers
include_directories(srcu)

add_executable(tentest
    srcu/tensor.cu
)

set_target_properties(tentest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    #CUDA_ARCHITECTURES "60"
    COMPILE_FLAGS "-g -DTENSOR_TEST_MAIN"
    LINK_FLAGS "-g"
)

add_executable(cumain
    srcu/main.cpp
    srcu/dataloader.cpp
    srcu/tensor.cu
)

set_target_properties(cumain PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    #CUDA_ARCHITECTURES "75"
    COMPILE_FLAGS "-g -DCUDA_AVAILABLE"
    LINK_FLAGS "-g"
)

target_link_libraries(cumain ${CUDA_LIBRARIES} curand)

# Add tensor tests executable
add_executable(tensor_tests
    srcu/tensor_tests.cu
    srcu/tensor.cu
)

set_target_properties(tensor_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "60"
    COMPILE_FLAGS "-g -DCUDA_AVAILABLE"
    LINK_FLAGS "-g"
)

target_link_libraries(tensor_tests ${CUDA_LIBRARIES} curand pthread)

# Include directories for headers
include_directories(src4)

# Create executable
add_executable(tests
    src4/tests.cpp
    src4/dataloader.cpp
)

add_executable(main
    srcu/main.cpp
    src4/dataloader.cpp
)