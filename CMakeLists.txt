cmake_minimum_required(VERSION 3.10)
project(cudann)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)

# Include directories for headers
include_directories(src4)

# Create executable
add_executable(tests
    src4/tests.cpp
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