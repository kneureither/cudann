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
set(CMAKE_CUDA_ARCHITECTURES "60")

# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g -O0 -fno-omit-frame-pointer)  # good stack traces

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # Debug flags for CUDA

# Include directories for headers
include_directories(srcu)

add_executable(cumain
    srcu/main.cpp
    srcu/dataloader.cpp
    srcu/tensor.cu
)

set_target_properties(cumain PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "60"
    COMPILE_FLAGS "-g"
    LINK_FLAGS "-g"
)

target_link_libraries(cumain ${CUDA_LIBRARIES} curand)

# Include directories for headers
include_directories(src4)

# Create executable
add_executable(tests
    src4/tests.cpp
    src4/dataloader.cpp
    # src4/model.cpp
    # Add any other .cpp files here
)

add_executable(main
    src4/main.cpp
    src4/dataloader.cpp
    # src4/model.cpp
    # Add any other .cpp files here
)

# Optional: if you want to create a library
# add_library(cudann_lib
#     src4/dataloader.cpp
#     src4/model.cpp
# )
# target_link_libraries(tests cudann_lib)