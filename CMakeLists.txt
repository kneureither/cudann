cmake_minimum_required(VERSION 3.10)
project(cudann)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g -O0 -fno-omit-frame-pointer)  # good stack traces

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

set_target_properties(main PROPERTIES
    COMPILE_FLAGS "-g"
    LINK_FLAGS "-g"
)

# Optional: if you want to create a library
# add_library(cudann_lib
#     src4/dataloader.cpp
#     src4/model.cpp
# )
# target_link_libraries(tests cudann_lib)