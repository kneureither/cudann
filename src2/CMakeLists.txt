cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the data directory path relative to the project root
set(DATA_DIR "${CMAKE_SOURCE_DIR}/dat")
add_compile_definitions(DATA_DIR="${DATA_DIR}")

# Source files for the core library
set(LIB_SOURCES
    neural_network.cpp
    tensor.cpp
    mnist_loader.cpp
)

# Header files
set(HEADERS
    neural_network.h
    mnist_loader.h
)

# Create the main library
add_library(nn_lib
    ${LIB_SOURCES}
    ${HEADERS}
)

# Set include directories for the library
target_include_directories(nn_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create the training executable
add_executable(mnist_trainer
    mnist-training-script.cpp
)

# Link the training executable with our library
target_link_libraries(mnist_trainer
    nn_lib
)

# Add a message to verify data directory during configuration
message(STATUS "MNIST data directory is set to: ${DATA_DIR}")

# Optional: Add a check to verify data directory exists
if(NOT EXISTS ${DATA_DIR})
    message(WARNING "Data directory ${DATA_DIR} does not exist. Please ensure MNIST data is placed in the correct location.") 