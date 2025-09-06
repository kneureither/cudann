#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

enum class Device
{
    CPU,
    GPU
};

// Forward declarations of templated CUDA kernels
template<typename T>
__global__ void zeros_kernel(T* data, size_t size);

template<typename T>
__global__ void ones_kernel(T* data, size_t size);

template<typename T>
__global__ void random_uniform_kernel(T* data, size_t size, T min_val, T max_val, unsigned long long seed);

template<typename T>
__global__ void add_tensors_kernel(T* a, const T* b, size_t size);

template<typename T>
__global__ void add_matrix_vector_kernel(T* matrix, const T* vector, size_t rows, size_t cols);

template<typename T>
__global__ void add_scalar_kernel(T* data, T scalar, size_t size);

template<typename T>
__global__ void subtract_tensors_kernel(T* a, const T* b, size_t size);

template<typename T>
__global__ void multiply_tensors_kernel(T* a, const T* b, size_t size);

template<typename T>
__global__ void multiply_scalar_kernel(T* data, T scalar, size_t size);

template<typename T>
__global__ void greater_than_kernel(const T* input, T* output, T threshold, size_t size);

template<typename T>
__global__ void matmul_kernel(const T* a, const T* b, T* c, size_t m, size_t n, size_t k);

template<typename T>
__global__ void transpose_kernel(const T* input, T* output, size_t rows, size_t cols);

template<typename T>
__global__ void exp_kernel(const T* input, T* output, size_t size);

template<typename T>
__global__ void log_kernel(const T* input, T* output, size_t size);

template<typename T>
__global__ void sum_axis0_kernel(const T* input, T* output, size_t rows, size_t cols);

template<typename T>
__global__ void sum_axis1_kernel(const T* input, T* output, size_t rows, size_t cols);

template<typename T>
__global__ void argmax_axis0_kernel(const T* input, int* output, size_t rows, size_t cols);

template<typename T>
__global__ void argmax_axis1_kernel(const T* input, int* output, size_t rows, size_t cols);

template<typename T>
__global__ void softmax_axis1_kernel(const T* input, T* output, size_t rows, size_t cols);

template<typename T>
__global__ void log_softmax_axis1_kernel(const T* input, T* output, size_t rows, size_t cols);

template <typename T>
__global__ void scatter_subtract_axis1_kernel(T* data, const int* indices, T value, size_t rows, size_t cols);

template<typename T>
__global__ void gather_axis1_kernel(const T* data, const int* indices, T* output, size_t rows, size_t cols);

template<typename T>
__global__ void update_value_kernel(T val, T* d_ptr);

template <typename T>
class Tensor
{
    // Allow different template instantiations to access each other's private members
    template <typename U> friend class Tensor;
private:
    T *data_ptr;               // Pointer to the data array (GPU memory)
    mutable T *cpu_data_ptr;   // CPU backup for debugging/printing
    size_t data_size;          // Total number of elements in the tensor
    bool owns_data;            // Flag to track ownership for proper cleanup
    Device device;             // Device the tensor is on
    Tensor *parent = nullptr; // keeps track for slicing
    std::vector<Tensor*> children;

    // Memory management methods
    void allocate_gpu_memory();
    void allocate_gpu_memory(std::vector<size_t> shape);
    void free_memory();

    // Helper function to calculate grid and block dimensions
    dim3 get_grid_size(size_t size, dim3 block_size = dim3(256)) const;
    dim3 get_grid_size_2d(size_t rows, size_t cols, dim3 block_size = dim3(16, 16)) const;

    // Copy data from GPU to CPU for operations that need CPU access
    void sync_to_cpu() const;

public:
    std::vector<size_t> shape;

    // Constructors and destructor
    Tensor();
    Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, Device device);
    Tensor(const Tensor &other);
    ~Tensor();

    // Assignment operator
    Tensor<T> &operator=(const Tensor<T> &other);

    // Initialization methods
    void zeros();
    void ones();
    void random_uniform(T min_val, T max_val);
    void update_value_at_idx(T val, size_t idx);

    // Getters
    std::vector<size_t> get_shape() const;
    size_t get_size() const;
    T *get_data_ptr(); // accesses device pointer
    const T *get_data_ptr() const; // accesses device pointer
    T *get_cpu_data_ptr(); // accesses cpu pointer
    const T *get_cpu_data_ptr() const; // accesses cpu pointer
    Device get_device() const;

    // Access operators (require CPU sync - use sparingly)
    T &operator[](int i);
    const T &operator[](int i) const;
    T &operator()(int i, int j);
    const T &operator()(int i, int j) const;
    Tensor<T> deepslice(size_t start_idx, size_t batch_size) const;
    Tensor<T> slice(size_t start_idx, size_t batch_size) const;
    void view(const Tensor<T> &other, size_t start_idx, size_t batch_size);

    // Arithmetic operators
    Tensor<T> &operator+=(const Tensor<T> &other);
    Tensor<T> &operator-=(const Tensor<T> &other);
    Tensor<T> &operator*=(const Tensor<T> &other);
    Tensor<T> &operator*=(const T &scalar);
    Tensor<T> operator>(const T &scalar) const;

    // Mathematical operations
    Tensor<T> matmul(const Tensor<T> &other) const;
    Tensor<T> transpose() const;
    Tensor<T> exp() const;
    Tensor<T> log() const;
    Tensor<T> sum(int axis) const;
    Tensor<int> argmax(int axis) const;
    Tensor<T> softmax_axis1() const;
    Tensor<T> log_softmax_axis1() const;
    void scatter_subtract_axis1(const Tensor<int>& indices, T value = T(1));
    Tensor<T> gather_axis1(const Tensor<int>& indices) const;

    // Device management and utilities
    void to_device(Device target_device);
    std::string to_string() const;
    std::string shape_to_string() const;
};

// Explicit template instantiation declaration
extern template class Tensor<float>;
extern template class Tensor<int>;

#endif
