#ifndef TENSOR_CU
#define TENSOR_CU

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

// CUDA kernels for basic operations
__global__ void zeros_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

__global__ void ones_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

__global__ void random_uniform_kernel(float* data, size_t size, float min_val, float max_val, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = min_val + curand_uniform(&state) * (max_val - min_val);
    }
}

__global__ void add_tensors_kernel(float* a, const float* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

__global__ void add_matrix_vector_kernel(float* matrix, const float* vector, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        matrix[idx] += vector[col];
    }
}

__global__ void add_scalar_kernel(float* data, float scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += scalar;
    }
}

__global__ void subtract_tensors_kernel(float* a, const float* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] -= b[idx];
    }
}

__global__ void multiply_tensors_kernel(float* a, const float* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

__global__ void multiply_scalar_kernel(float* data, float scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scalar;
    }
}

__global__ void greater_than_kernel(const float* input, float* output, float threshold, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > threshold) ? 1.0f : 0.0f;
    }
}

__global__ void matmul_kernel(const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void transpose_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void exp_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

__global__ void log_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

__global__ void sum_axis0_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        float sum = 0.0f;
        for (size_t row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

__global__ void sum_axis1_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}

__global__ void argmax_axis0_kernel(const float* input, int* output, size_t rows, size_t cols) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        size_t best_idx = 0;
        float best_val = input[col];
        
        for (size_t row = 1; row < rows; row++) {
            float val = input[row * cols + col];
            if (val > best_val) {
                best_val = val;
                best_idx = row;
            }
        }
        output[col] = static_cast<int>(best_idx);
    }
}

__global__ void argmax_axis1_kernel(const float* input, int* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        size_t best_idx = 0;
        float best_val = input[row * cols];
        
        for (size_t col = 1; col < cols; col++) {
            float val = input[row * cols + col];
            if (val > best_val) {
                best_val = val;
                best_idx = col;
            }
        }
        output[row] = static_cast<int>(best_idx);
    }
}

__global__ void softmax_axis1_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find row max
        float max_val = input[row * cols];
        for (size_t col = 1; col < cols; col++) {
            float val = input[row * cols + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float exp_val = expf(input[row * cols + col] - max_val);
            output[row * cols + col] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (size_t col = 0; col < cols; col++) {
            output[row * cols + col] /= sum_exp;
        }
    }
}

__global__ void log_softmax_axis1_kernel(const float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find row max
        float max_val = input[row * cols];
        for (size_t col = 1; col < cols; col++) {
            float val = input[row * cols + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Compute log-sum-exp
        float sum_exp = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            sum_exp += expf(input[row * cols + col] - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);
        
        // Compute log-softmax
        for (size_t col = 0; col < cols; col++) {
            output[row * cols + col] = input[row * cols + col] - log_sum_exp;
        }
    }
}

template <typename T>
class Tensor
{
private:
    T *data_ptr;               // Pointer to the data array (GPU memory)
    T *cpu_data_ptr;           // CPU backup for debugging/printing
    size_t data_size;          // Total number of elements in the tensor
    bool owns_data;            // Flag to track ownership for proper cleanup
    Device device;             // Device the tensor is on

    // Memory management methods
    void allocate_gpu_memory()
    {
        if (data_size > 0)
        {
            CUDA_CHECK(cudaMalloc(&data_ptr, data_size * sizeof(T)));
            owns_data = true;
            device = Device::GPU;
        }
    }

    void allocate_gpu_memory(std::vector<size_t> shape)
    {
        this->shape = shape;
        data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        logger("allocate_gpu_memory(shape) data size: " + std::to_string(data_size), "DEBUG", __FILE__, __LINE__);
        allocate_gpu_memory();
    }

    // Free the allocated memory
    void free_memory()
    {
        if (owns_data && data_ptr != nullptr)
        {
            logger("freeing GPU memory", "DEBUG", __FILE__, __LINE__);
            CUDA_CHECK(cudaFree(data_ptr));
            data_ptr = nullptr;
        }
        if (cpu_data_ptr != nullptr) {
            delete[] cpu_data_ptr;
            cpu_data_ptr = nullptr;
        }
        owns_data = false;
        this->data_size = 0;
        this->shape = std::vector<size_t>({});
    }

    // Helper function to calculate grid and block dimensions
    dim3 get_grid_size(size_t size, dim3 block_size = dim3(256)) const {
        return dim3((size + block_size.x - 1) / block_size.x);
    }

    dim3 get_grid_size_2d(size_t rows, size_t cols, dim3 block_size = dim3(16, 16)) const {
        return dim3((cols + block_size.x - 1) / block_size.x, 
                   (rows + block_size.y - 1) / block_size.y);
    }

    // Copy data from GPU to CPU for operations that need CPU access
    void sync_to_cpu() const {
        if (device == Device::GPU && data_ptr != nullptr) {
            if (cpu_data_ptr == nullptr) {
                cpu_data_ptr = new T[data_size];
            }
            CUDA_CHECK(cudaMemcpy(cpu_data_ptr, data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

public:
    std::vector<size_t> shape;
    mutable T *cpu_data_ptr;  // Make mutable for const methods that need CPU access

    // Default constructor
    Tensor() : data_ptr(nullptr), cpu_data_ptr(nullptr), data_size(0), owns_data(false), device(Device::GPU) {}

    // Constructor for Tensor with specified shape (defaults to GPU)
    Tensor(std::vector<size_t> shape) : shape(shape), data_size(0), owns_data(false), cpu_data_ptr(nullptr)
    {
        allocate_gpu_memory(shape);
    }

    Tensor(std::vector<size_t> shape, Device device) : shape(shape), data_size(0), owns_data(false), device(device), cpu_data_ptr(nullptr)
    {
        if (device == Device::GPU) {
            allocate_gpu_memory(shape);
        } else {
            throw std::runtime_error("CPU-only tensors not supported in CUDA version");
        }
    }

    // Assignment operator
    Tensor<T> &operator=(const Tensor<T> &other)
    {
        if (this != &other)
        {
            logger("assigning new tensor, old shape " + this->shape_to_string() + " other shape: " + other.shape_to_string() +
                       " this owns data " + std::to_string(this->owns_data),
                   "DEBUG", __FILE__, __LINE__);
            
            if (this->owns_data)
            {
                logger("freeing memory", "DEBUG", __FILE__, __LINE__);
                free_memory();
            }
            shape = other.shape;
            data_size = other.data_size;
            device = other.device;
            allocate_gpu_memory(shape);
            
            // Copy data on GPU
            CUDA_CHECK(cudaMemcpy(data_ptr, other.data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToDevice));
            owns_data = true;
        }
        return *this;
    }

    // Copy constructor
    Tensor(const Tensor &other) : shape(other.shape), data_size(other.data_size), owns_data(false), device(other.device), cpu_data_ptr(nullptr)
    {
        allocate_gpu_memory(other.shape);
        CUDA_CHECK(cudaMemcpy(data_ptr, other.data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void zeros()
    {
        if (data_size > 0) {
            dim3 block_size(256);
            dim3 grid_size = get_grid_size(data_size, block_size);
            zeros_kernel<<<grid_size, block_size>>>(data_ptr, data_size);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    void ones()
    {
        if (data_size > 0) {
            dim3 block_size(256);
            dim3 grid_size = get_grid_size(data_size, block_size);
            ones_kernel<<<grid_size, block_size>>>(data_ptr, data_size);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    void random_uniform(T min_val, T max_val)
    {
        if (data_size > 0) {
            dim3 block_size(256);
            dim3 grid_size = get_grid_size(data_size, block_size);
            unsigned long long seed = time(NULL);
            random_uniform_kernel<<<grid_size, block_size>>>(data_ptr, data_size, min_val, max_val, seed);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Destructor for Tensor
    ~Tensor()
    {
        free_memory();
    }

    // Getters for shape and data size
    std::vector<size_t> get_shape() const { return shape; }
    size_t get_size() const { return data_size; }

    // Getters for data pointer (useful for CUDA operations)
    T *get_data_ptr() { return data_ptr; }
    const T *get_data_ptr() const { return data_ptr; }

    //// ACCESS OPERATORS ////
    // Note: These require CPU sync and should be used sparingly

    T &operator[](int i)
    {
        if (shape.size() != 1)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 1D tensor");
        }
        if (i < 0 || i >= data_size)
        {
            throw std::out_of_range("Index out of bounds: " + std::to_string(i));
        }
        sync_to_cpu();
        return cpu_data_ptr[i];
    }

    const T &operator[](int i) const
    {
        if (shape.size() != 1)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 1D tensor");
        }
        if (i < 0 || static_cast<size_t>(i) >= data_size)
        {
            throw std::out_of_range("Index out of bounds");
        }
        sync_to_cpu();
        return cpu_data_ptr[i];
    }

    T &operator()(int i, int j)
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 2D tensor");
        }
        if (i < 0 || j < 0 || static_cast<size_t>(i) >= shape[0] || static_cast<size_t>(j) >= shape[1])
        {
            throw std::out_of_range("Index out of bounds");
        }
        sync_to_cpu();
        return cpu_data_ptr[static_cast<size_t>(i) * shape[1] + static_cast<size_t>(j)];
    }

    const T &operator()(int i, int j) const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 2D tensor");
        }
        if (i < 0 || j < 0 || static_cast<size_t>(i) >= shape[0] || static_cast<size_t>(j) >= shape[1])
        {
            throw std::out_of_range("Index out of bounds");
        }
        sync_to_cpu();
        return cpu_data_ptr[static_cast<size_t>(i) * shape[1] + static_cast<size_t>(j)];
    }

    //// OTHER OPERATORS ////

    Tensor<T> &operator+=(const Tensor<T> &other)
    {
        if (shape == other.shape)
        {
            logger("Adding tensors with shape: " + shape_to_string() + " and " + other.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            dim3 block_size(256);
            dim3 grid_size = get_grid_size(data_size, block_size);
            add_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
            CUDA_CHECK(cudaGetLastError());
            return *this;
        } 
        // adding a vector to matrix (e.g. [batch, features] + [features])
        else if (shape.size() == 2 && other.shape.size() == 1 && shape[1] == other.shape[0])
        {
            logger("Adding matrix with shape: " + shape_to_string() + " and vector with shape: " + other.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            dim3 block_size(16, 16);
            dim3 grid_size = get_grid_size_2d(shape[0], shape[1], block_size);
            add_matrix_vector_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, shape[0], shape[1]);
            CUDA_CHECK(cudaGetLastError());
            return *this;
        }
        // adding a scalar to the tensor
        else if (other.shape.size() == 1 && other.shape[0] == 1 && shape.size() > 0)
        {   
            other.sync_to_cpu();
            logger("Adding scalar to tensor with shape: " + shape_to_string() + " and scalar: " + std::to_string(other.cpu_data_ptr[0]), "DEBUG", __FILE__, __LINE__);
            dim3 block_size(256);
            dim3 grid_size = get_grid_size(data_size, block_size);
            add_scalar_kernel<<<grid_size, block_size>>>(data_ptr, other.cpu_data_ptr[0], data_size);
            CUDA_CHECK(cudaGetLastError());
            return *this;
        }
        else
        {
            throw std::invalid_argument("Tried to add tensors with different shapes: " + shape_to_string() + " and " + other.shape_to_string());
        }
    }

    Tensor<T> &operator-=(const Tensor<T> &other)
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tried to subtract tensors with different shapes");
        }
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        subtract_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor<T> &operator*=(const Tensor<T> &other)
    {
        if (shape != other.shape) throw std::invalid_argument("Tried to multiply tensors with different shapes");
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        multiply_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor<T> &operator*=(const T &scalar)
    {
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        multiply_scalar_kernel<<<grid_size, block_size>>>(data_ptr, scalar, data_size);
        CUDA_CHECK(cudaGetLastError());
        return *this;
    }

    Tensor<T> operator>(const T &scalar) const {
        Tensor<T> result(shape);
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        greater_than_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, scalar, data_size);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    //// MATHEMATICAL OPERATIONS ////

    Tensor<T> matmul(const Tensor<T> &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("Tried to matrix multiply tensors with shape " + shape_to_string() + " and " + other.shape_to_string());
        }

        if (shape[1] != other.shape[0])
        {
            throw std::invalid_argument("Tried to matrix multiply tensors with shape " + shape_to_string() + " and " + other.shape_to_string());
        }

        Tensor<T> result({shape[0], other.shape[1]});
        
        dim3 block_size(16, 16);
        dim3 grid_size = get_grid_size_2d(shape[0], other.shape[1], block_size);
        matmul_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, result.data_ptr, 
                                                shape[0], other.shape[1], shape[1]);
        CUDA_CHECK(cudaGetLastError());

        return result;
    }

    Tensor<T> transpose() const
    {
        if (shape.size() == 1)
        {
            // 1D tensor transpose is just a copy
            return *this;
        }
        else if (shape.size() == 2)
        {
            Tensor<T> result({shape[1], shape[0]});
            dim3 block_size(16, 16);
            dim3 grid_size = get_grid_size_2d(shape[0], shape[1], block_size);
            transpose_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, shape[0], shape[1]);
            CUDA_CHECK(cudaGetLastError());
            return result;
        }
        else
        {
            throw std::invalid_argument("Transpose not implemented for tensors with more than 2 dimensions");
        }
    }

    Tensor<T> exp() const
    {
        Tensor<T> result(shape);
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        exp_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    Tensor<T> log() const
    {
        Tensor<T> result(shape);
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        log_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    Tensor<T> sum(int axis) const
    {
        if (axis < 0 || axis >= shape.size())
        {
            throw std::invalid_argument("Axis out of bounds");
        }
        
        if (shape.size() == 1)
        {
            // For 1D tensor, sum all elements
            sync_to_cpu();
            T sum_val = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                sum_val += cpu_data_ptr[i];
            }
            Tensor<T> result({1});
            CUDA_CHECK(cudaMemcpy(result.data_ptr, &sum_val, sizeof(T), cudaMemcpyHostToDevice));
            return result;
        }
        else if (shape.size() == 2)
        {
            if (axis == 0)
            {
                // sum along rows → [cols]
                Tensor<T> result({shape[1]});
                dim3 block_size(256);
                dim3 grid_size = get_grid_size(shape[1], block_size);
                sum_axis0_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, shape[0], shape[1]);
                CUDA_CHECK(cudaGetLastError());
                return result;
            }
            else if (axis == 1)
            {
                // sum along columns → [rows]
                Tensor<T> result({shape[0]});
                dim3 block_size(256);
                dim3 grid_size = get_grid_size(shape[0], block_size);
                sum_axis1_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, shape[0], shape[1]);
                CUDA_CHECK(cudaGetLastError());
                return result;
            } else {
                throw std::invalid_argument("sum(): axis must be 0 or 1, was " + std::to_string(axis));
            }
        }
        else {
            throw std::invalid_argument("Sum not implemented for tensors with more than 2 dimensions");
        }
        return Tensor<T>();
    }

    Tensor<int> argmax(int axis) const
    {
        if (shape.size() == 1) {
            sync_to_cpu();
            Tensor<int> out({1});
            size_t best_i = 0;
            T best_v = cpu_data_ptr[0];
            for (size_t i = 1; i < shape[0]; i++) {
                T v = cpu_data_ptr[i];
                if (v > best_v)
                {
                    best_v = v;
                    best_i = i;
                }
            }
            int result = static_cast<int>(best_i);
            CUDA_CHECK(cudaMemcpy(out.data_ptr, &result, sizeof(int), cudaMemcpyHostToDevice));
            return out;
        } else if (shape.size() == 2) {
            if (axis == 0)
            {
                // argmax along rows → [cols]
                Tensor<int> out({shape[1]});
                dim3 block_size(256);
                dim3 grid_size = get_grid_size(shape[1], block_size);
                argmax_axis0_kernel<<<grid_size, block_size>>>(data_ptr, out.data_ptr, shape[0], shape[1]);
                CUDA_CHECK(cudaGetLastError());
                return out;
            }
            else if (axis == 1)
            {
                // argmax along cols → [rows]
                Tensor<int> out({shape[0]});
                dim3 block_size(256);
                dim3 grid_size = get_grid_size(shape[0], block_size);
                argmax_axis1_kernel<<<grid_size, block_size>>>(data_ptr, out.data_ptr, shape[0], shape[1]);
                CUDA_CHECK(cudaGetLastError());
                return out;
            }
            else
            {
                throw std::invalid_argument("argmax: axis must be 0 or 1");
            }
        } else {
            throw std::invalid_argument("argmax: 1D or 2D only");
        }
    }

    Tensor<T> softmax_axis1() const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("softmax_axis1 expects a 2D tensor");
        }

        Tensor<T> out({shape[0], shape[1]});
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(shape[0], block_size);
        softmax_axis1_kernel<<<grid_size, block_size>>>(data_ptr, out.data_ptr, shape[0], shape[1]);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Tensor<T> log_softmax_axis1() const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("log_softmax_axis1 expects a 2D tensor");
        }

        Tensor<T> out({shape[0], shape[1]});
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(shape[0], block_size);
        log_softmax_axis1_kernel<<<grid_size, block_size>>>(data_ptr, out.data_ptr, shape[0], shape[1]);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    //// DEVICE MANAGEMENT AND UTILS ////

    void to_device(Device target_device)
    {
        if (target_device == this->device)
        {
            // Already on target device
            return;
        }
        // Note: In this CUDA-only implementation, we always stay on GPU
        // CPU transfers are only for debugging/printing via sync_to_cpu()
        throw std::runtime_error("Device transfer not implemented in CUDA-only version");
    }

    Device get_device() const { return device; }

    std::string to_string() const
    {
        sync_to_cpu();
        std::stringstream ss;
        if (shape.size() == 1)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << cpu_data_ptr[i];
                if (i < shape[0] - 1) ss << ", ";
            }
            ss << "]";
        }
        else if (shape.size() == 2)
        {
            ss << "[\n";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << "  [";
                for (int j = 0; j < shape[1]; j++)
                {
                    ss << cpu_data_ptr[i * shape[1] + j];
                    if (j < shape[1] - 1) ss << ", ";
                }
                ss << "]";
                if (i < shape[0] - 1) ss << ",";
                ss << "\n";
            }
            ss << "]";
        }
        else
        {
            throw std::invalid_argument("to_string not implemented for tensors with more than 2 dimensions");
        }
        return ss.str();
    }

    std::string shape_to_string() const
    {
        if (shape.size() == 0)
            return std::string("( null )");

        std::stringstream ss;
        ss << "( ";
        for (int i = 0; i < shape.size() - 1; i++)
        {
            ss << shape[i] << ", ";
        }
        ss << shape[shape.size() - 1] << " )";
        return ss.str();
    }
};

// Explicit template instantiation for float
template class Tensor<float>;

#endif
