#include "tensor.cuh"
#include <ctime>

// Templated CUDA kernels
template<typename T>
__global__ void zeros_kernel(T* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = T(0);
    }
}

template<typename T>
__global__ void ones_kernel(T* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = T(1);
    }
}

template<typename T>
__global__ void random_uniform_kernel(T* data, size_t size, T min_val, T max_val, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        if constexpr (std::is_same_v<T, float>) {
            data[idx] = min_val + curand_uniform(&state) * (max_val - min_val);
        } else if constexpr (std::is_same_v<T, int>) {
            data[idx] = min_val + static_cast<int>(curand_uniform(&state) * (max_val - min_val + 1));
        }
    }
}

template<typename T>
__global__ void add_tensors_kernel(T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

// todo needs improvement. start one kernel per vector elem? also check the block dims
template<typename T>
__global__ void add_matrix_vector_kernel(T* matrix, const T* vector, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        matrix[idx] += vector[col];
    }
}

template<typename T>
__global__ void add_scalar_kernel(T* data, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += scalar;
    }
}

template<typename T>
__global__ void subtract_tensors_kernel(T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] -= b[idx];
    }
}

template<typename T>
__global__ void multiply_tensors_kernel(T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

template<typename T>
__global__ void multiply_scalar_kernel(T* data, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scalar;
    }
}

template<typename T>
__global__ void greater_than_kernel(const T* input, T* output, T threshold, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > threshold) ? T(1) : T(0);
    }
}

// needs some love
template<typename T>
__global__ void matmul_kernel(const T* a, const T* b, T* c, size_t m, size_t n, size_t k) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        T sum = T(0);
        for (size_t i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

template<typename T>
__global__ void transpose_kernel(const T* input, T* output, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Math functions - only for float types
template<typename T>
__global__ void exp_kernel(const T* input, T* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float>) {
            output[idx] = expf(input[idx]);
        } else {
            // For integer types, exp doesn't make sense, so we'll just copy
            output[idx] = input[idx];
        }
    }
}

template<typename T>
__global__ void log_kernel(const T* input, T* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float>) {
            output[idx] = logf(input[idx]);
        } else {
            // For integer types, log doesn't make sense, so we'll just copy
            output[idx] = input[idx];
        }
    }
}

template<typename T>
__global__ void sum_axis0_kernel(const T* input, T* output, size_t rows, size_t cols) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        T sum = T(0);
        for (size_t row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

template<typename T>
__global__ void sum_axis1_kernel(const T* input, T* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        T sum = T(0);
        for (size_t col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}

template<typename T>
__global__ void argmax_axis0_kernel(const T* input, int* output, size_t rows, size_t cols) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        size_t best_idx = 0;
        T best_val = input[col];
        
        for (size_t row = 1; row < rows; row++) {
            T val = input[row * cols + col];
            if (val > best_val) {
                best_val = val;
                best_idx = row;
            }
        }
        output[col] = static_cast<int>(best_idx);
    }
}

template<typename T>
__global__ void argmax_axis1_kernel(const T* input, int* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        size_t best_idx = 0;
        T best_val = input[row * cols];
        
        for (size_t col = 1; col < cols; col++) {
            T val = input[row * cols + col];
            if (val > best_val) {
                best_val = val;
                best_idx = col;
            }
        }
        output[row] = static_cast<int>(best_idx);
    }
}

// Softmax kernels - only meaningful for float
template<typename T>
__global__ void softmax_axis1_kernel(const T* input, T* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        if constexpr (std::is_same_v<T, float>) {
            // Find row max
            T max_val = input[row * cols];
            for (size_t col = 1; col < cols; col++) {
                T val = input[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // Compute sum of exp(x - max)
            T sum_exp = T(0);
            for (size_t col = 0; col < cols; col++) {
                T exp_val = expf(input[row * cols + col] - max_val);
                output[row * cols + col] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (size_t col = 0; col < cols; col++) {
                output[row * cols + col] /= sum_exp;
            }
        } else {
            // For non-float types, just copy input to output
            for (size_t col = 0; col < cols; col++) {
                output[row * cols + col] = input[row * cols + col];
            }
        }
    }
}

template<typename T>
__global__ void log_softmax_axis1_kernel(const T* input, T* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        if constexpr (std::is_same_v<T, float>) {
            // Find row max
            T max_val = input[row * cols];
            for (size_t col = 1; col < cols; col++) {
                T val = input[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // Compute log-sum-exp
            T sum_exp = T(0);
            for (size_t col = 0; col < cols; col++) {
                sum_exp += expf(input[row * cols + col] - max_val);
            }
            T log_sum_exp = max_val + logf(sum_exp);
            
            // Compute log-softmax
            for (size_t col = 0; col < cols; col++) {
                output[row * cols + col] = input[row * cols + col] - log_sum_exp;
            }
        } else {
            // For non-float types, just copy input to output
            for (size_t col = 0; col < cols; col++) {
                output[row * cols + col] = input[row * cols + col];
            }
        }
    }
}

template<typename T>
__global__ void scatter_subtract_axis1_kernel(T* data, const int* indices, T value, size_t rows, size_t cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        int col_idx = indices[row];
        if (col_idx >= 0 && col_idx < cols) {
            data[row * cols + col_idx] -= value;
        }
    }
}

template <typename T>
__global__ void update_value_kernel(T val, T* d_ptr) {
    //cudaMemcpy(d_ptr, &val, sizeof(T), cudaMemcpyHostToDevice);
    *d_ptr = val;
}

// Tensor class implementation

template <typename T>
void Tensor<T>::allocate_gpu_memory()
{
    if (data_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&data_ptr, data_size * sizeof(T)));

        owns_data = true;
        device = Device::GPU;
    }
}

template <typename T>
void Tensor<T>::allocate_gpu_memory(std::vector<size_t> shape)
{
    this->shape = shape;
    data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    logger("allocate_gpu_memory(shape) data size: " + std::to_string(data_size), "DEBUG", __FILE__, __LINE__);
    allocate_gpu_memory();
}

template <typename T>
void Tensor<T>::free_memory()
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

template <typename T>
void Tensor<T>::update_value_at_idx(T val, size_t idx) {
    if(idx < data_size) {
        T* elem_ptr = data_ptr + idx;
        update_value_kernel<T><<<1, 1>>>(val, elem_ptr);
    } else {
        throw std::out_of_range("Index out of bounds: " + std::to_string(idx));
    }
}

template <typename T>
dim3 Tensor<T>::get_grid_size(size_t size, dim3 block_size) const {
    return dim3((size + block_size.x - 1) / block_size.x);
}

template <typename T>
dim3 Tensor<T>::get_grid_size_2d(size_t rows, size_t cols, dim3 block_size) const {
    return dim3((cols + block_size.x - 1) / block_size.x, 
               (rows + block_size.y - 1) / block_size.y);
}

template <typename T>
void Tensor<T>::sync_to_cpu() const {
    if (device == Device::GPU && data_ptr != nullptr) {
        if (cpu_data_ptr == nullptr) {
            cpu_data_ptr = new T[data_size];
        }
        CUDA_CHECK(cudaMemcpy(cpu_data_ptr, data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template <typename T>
Tensor<T>::Tensor() : data_ptr(nullptr), cpu_data_ptr(nullptr), data_size(0), owns_data(false), device(Device::GPU) {}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : shape(shape), data_size(0), owns_data(false), cpu_data_ptr(nullptr)
{
    allocate_gpu_memory(shape);
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, Device device) : shape(shape), data_size(0), owns_data(false), device(device), cpu_data_ptr(nullptr)
{
    if (device == Device::GPU) {
        allocate_gpu_memory(shape);
    } else {
        throw std::runtime_error("CPU-only tensors not supported in CUDA version");
    }
}

template <typename T>
Tensor<T>::Tensor(const Tensor &other) : shape(other.shape), data_size(other.data_size), owns_data(false), device(other.device), cpu_data_ptr(nullptr)
{
    allocate_gpu_memory(other.shape);
    CUDA_CHECK(cudaMemcpy(data_ptr, other.data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
Tensor<T>::~Tensor()
{
    free_memory();
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
{
    if (this != &other)
    {
        logger("assigning new tensor, old shape " + this->shape_to_string() + " other shape: " + other.shape_to_string() +
                   " this owns data " + std::to_string(this->owns_data),
               "DEBUG", __FILE__, __LINE__);

        if (this-> owns_data && this->data_size == other.data_size && this->device == other.device) {
            shape = other.shape;
        } else {
            if (this->owns_data)
            {
                logger("freeing memory", "DEBUG", __FILE__, __LINE__);
                free_memory();
            }
            shape = other.shape;
            data_size = other.data_size;
            device = other.device;
            allocate_gpu_memory(shape);
        }
        
        // Copy data on GPU
        CUDA_CHECK(cudaMemcpy(data_ptr, other.data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToDevice));
        owns_data = true;
    }
    return *this;
}

template <typename T>
void Tensor<T>::zeros()
{
    if (data_size > 0) {
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        zeros_kernel<T><<<grid_size, block_size>>>(data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
void Tensor<T>::ones()
{
    if (data_size > 0) {
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        ones_kernel<T><<<grid_size, block_size>>>(data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
void Tensor<T>::random_uniform(T min_val, T max_val)
{
    if (data_size > 0) {
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        unsigned long long seed = time(NULL);
        random_uniform_kernel<T><<<grid_size, block_size>>>(data_ptr, data_size, min_val, max_val, seed);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
std::vector<size_t> Tensor<T>::get_shape() const { return shape; }

template <typename T>
size_t Tensor<T>::get_size() const { return data_size; }

template <typename T>
T *Tensor<T>::get_data_ptr() { return data_ptr; }

template <typename T>
const T *Tensor<T>::get_data_ptr() const { return data_ptr; }

template <typename T>
T *Tensor<T>::get_cpu_data_ptr() { sync_to_cpu(); return cpu_data_ptr; }

template <typename T>
const T *Tensor<T>::get_cpu_data_ptr() const { sync_to_cpu(); return cpu_data_ptr; }

template <typename T>
Device Tensor<T>::get_device() const { return device; }

// Access operators implementation
template <typename T>
T &Tensor<T>::operator[](int i)
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

template <typename T>
const T &Tensor<T>::operator[](int i) const
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

template <typename T>
T &Tensor<T>::operator()(int i, int j)
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

template <typename T>
const T &Tensor<T>::operator()(int i, int j) const
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

template <typename T>
Tensor<T> Tensor<T>::deepslice(size_t start_idx, size_t batch_size) const {
    //logger("Slicing used, start_idx" + std::to_string(start_idx) + ", batch " + std::to_string(batch_size), "INFO", __FILE__, __LINE__);
    if (shape.size() == 2) {
    
    size_t N = shape[0];
    size_t C = shape[1];
    
    // Validate indices
    if (start_idx >= N) {
        throw std::out_of_range("slice: start_idx (" + std::to_string(start_idx) + 
                               ") >= tensor size (" + std::to_string(N) + ")");
    }
    
    if (start_idx + batch_size > N) {
        throw std::out_of_range("slice: start_idx + batch_size (" + 
                               std::to_string(start_idx + batch_size) + 
                               ") > tensor size (" + std::to_string(N) + ")");
    }
    
    // Create result tensor
    Tensor<T> result({batch_size, C});
    
    // Use cudaMemcpy for efficient contiguous memory copy
    const T* src_ptr = data_ptr + (start_idx * C);
    T* dst_ptr = result.get_data_ptr();
    size_t copy_size = batch_size * C * sizeof(T);
    
    CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice));
    
    return result;

    } else if (shape.size() == 1) {
          
        size_t N = shape[0];
        
        // Validate indices
        if (start_idx >= N) {
            throw std::out_of_range("slice: start_idx (" + std::to_string(start_idx) + 
                                ") >= tensor size (" + std::to_string(N) + ")");
        }
        
        if (start_idx + batch_size > N) {
            throw std::out_of_range("slice: start_idx + batch_size (" + 
                                std::to_string(start_idx + batch_size) + 
                                ") > tensor size (" + std::to_string(N) + ")");
        }
        // Create result tensor
        Tensor<T> result({batch_size});
        
        // Use cudaMemcpy for efficient contiguous memory copy
        const T* src_ptr = data_ptr + (start_idx);
        T* dst_ptr = result.get_data_ptr();
        size_t copy_size = batch_size * sizeof(T);
        
        CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice));
        
        return result;
    } else {
        throw std::invalid_argument("slice: tensor must be 1D [N] or 2D [N, C], got shape " + shape_to_string());
    }
}

// Arithmetic operators implementation
template <typename T>
Tensor<T> &Tensor<T>::operator+=(const Tensor<T> &other)
{
    if (shape == other.shape)
    {
        logger("Adding tensors with shape: " + shape_to_string() + " and " + other.shape_to_string(), "DEBUG", __FILE__, __LINE__);
        dim3 block_size(256);
        dim3 grid_size = get_grid_size(data_size, block_size);
        add_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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
        CUDA_CHECK(cudaDeviceSynchronize());
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
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }
    else
    {
        throw std::invalid_argument("Tried to add tensors with different shapes: " + shape_to_string() + " and " + other.shape_to_string());
    }
}

template <typename T>
Tensor<T> &Tensor<T>::operator-=(const Tensor<T> &other)
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Tried to subtract tensors with different shapes");
    }
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    subtract_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=(const Tensor<T> &other)
{
    if (shape != other.shape) throw std::invalid_argument("Tried to multiply tensors with different shapes");
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    multiply_tensors_kernel<<<grid_size, block_size>>>(data_ptr, other.data_ptr, data_size);
    CUDA_CHECK(cudaGetLastError());
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=(const T &scalar)
{
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    multiply_scalar_kernel<<<grid_size, block_size>>>(data_ptr, scalar, data_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator>(const T &scalar) const {
    Tensor<T> result(shape);
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    greater_than_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, scalar, data_size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// Mathematical operations implementation
template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> &other) const
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
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const
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

template <typename T>
Tensor<T> Tensor<T>::exp() const
{
    Tensor<T> result(shape);
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    exp_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, data_size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::log() const
{
    Tensor<T> result(shape);
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(data_size, block_size);
    log_kernel<<<grid_size, block_size>>>(data_ptr, result.data_ptr, data_size);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::sum(int axis) const
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

template <typename T>
Tensor<int> Tensor<T>::argmax(int axis) const
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

template <typename T>
Tensor<T> Tensor<T>::softmax_axis1() const
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

template <typename T>
Tensor<T> Tensor<T>::log_softmax_axis1() const
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


template <typename T>
void Tensor<T>::scatter_subtract_axis1(const Tensor<int>& indices, T value) {
    if (shape.size() != 2) {
        throw std::invalid_argument("scatter_subtract_axis1: tensor must be 2D");
    }
    if (indices.shape.size() != 1) {
        throw std::invalid_argument("scatter_subtract_axis1: indices must be 1D");
    }
    if (indices.shape[0] != shape[0]) {
        throw std::invalid_argument("scatter_subtract_axis1: indices length must match number of rows");
    }
    
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    dim3 block_size(256);
    dim3 grid_size = get_grid_size(rows, block_size);
    
    scatter_subtract_axis1_kernel<<<grid_size, block_size>>>(
        data_ptr, indices.get_data_ptr(), value, rows, cols
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void Tensor<T>::to_device(Device target_device)
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

template <typename T>
std::string Tensor<T>::to_string() const
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

template <typename T>
std::string Tensor<T>::shape_to_string() const
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

// Explicit template instantiation for float
template class Tensor<float>;
template class Tensor<int>;

// Explicit kernel instantiations
template __global__ void zeros_kernel<float>(float*, size_t);
template __global__ void zeros_kernel<int>(int*, size_t);
template __global__ void ones_kernel<float>(float*, size_t);
template __global__ void ones_kernel<int>(int*, size_t);
template __global__ void add_tensors_kernel<float>(float*, const float*, size_t);
template __global__ void add_tensors_kernel<int>(int*, const int*, size_t);
template __global__ void random_uniform_kernel<float>(float*, size_t, float, float, unsigned long long);
template __global__ void random_uniform_kernel<int>(int*, size_t, int, int, unsigned long long);
template __global__ void add_matrix_vector_kernel<float>(float*, const float*, size_t, size_t);
template __global__ void add_matrix_vector_kernel<int>(int*, const int*, size_t, size_t);
template __global__ void add_scalar_kernel<float>(float*, float, size_t);
template __global__ void add_scalar_kernel<int>(int*, int, size_t);
template __global__ void subtract_tensors_kernel<float>(float*, const float*, size_t);
template __global__ void subtract_tensors_kernel<int>(int*, const int*, size_t);
template __global__ void multiply_tensors_kernel<float>(float*, const float*, size_t);
template __global__ void multiply_tensors_kernel<int>(int*, const int*, size_t);
template __global__ void multiply_scalar_kernel<float>(float*, float, size_t);
template __global__ void multiply_scalar_kernel<int>(int*, int, size_t);
template __global__ void greater_than_kernel<float>(const float*, float*, float, size_t);
template __global__ void greater_than_kernel<int>(const int*, int*, int, size_t);
template __global__ void matmul_kernel<float>(const float*, const float*, float*, size_t, size_t, size_t);
template __global__ void matmul_kernel<int>(const int*, const int*, int*, size_t, size_t, size_t);
template __global__ void transpose_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void transpose_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void exp_kernel<float>(const float*, float*, size_t);
template __global__ void exp_kernel<int>(const int*, int*, size_t);
template __global__ void log_kernel<float>(const float*, float*, size_t);
template __global__ void log_kernel<int>(const int*, int*, size_t);
template __global__ void sum_axis0_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void sum_axis0_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void sum_axis1_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void sum_axis1_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void argmax_axis0_kernel<float>(const float*, int*, size_t, size_t);
template __global__ void argmax_axis0_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void argmax_axis1_kernel<float>(const float*, int*, size_t, size_t);
template __global__ void argmax_axis1_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void softmax_axis1_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void softmax_axis1_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void log_softmax_axis1_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void log_softmax_axis1_kernel<int>(const int*, int*, size_t, size_t);
template __global__ void scatter_subtract_axis1_kernel<float>(float*, const int*, float, size_t, size_t);
template __global__ void scatter_subtract_axis1_kernel<int>(int*, const int*, int, size_t, size_t);
template __global__ void update_value_kernel<float>(float, float*);
template __global__ void update_value_kernel<int>(int, int*);

// Test main function (only compiled when this file is used as main)
#ifdef TENSOR_TEST_MAIN
int main() {
    std::cout << "Testing CUDA Tensor implementation..." << std::endl;
    
    try {
        // Test basic tensor creation
        Tensor<float> a({3, 4});
        std::cout << "Created tensor with shape: " << a.shape_to_string() << std::endl;
        
        // Test initialization
        a.ones();
        std::cout << "Initialized tensor with ones" << std::endl;
        std::cout << a.to_string() << std::endl << std::endl;
        
        // Test another tensor
        Tensor<float> b({3, 4});
        b.random_uniform(-1.0f, 1.0f);
        std::cout << "Created random tensor b:" << std::endl;
        std::cout << b.to_string() << std::endl << std::endl;
        
        // Test addition
        a += b;
        std::cout << "Performed tensor addition, tensor a:" << std::endl;
        std::cout << a.to_string() << std::endl << std::endl;
        
        // Test matrix multiplication
        Tensor<float> c({4, 2});
        c.ones();
        std::cout <<"Tensor c:" << std::endl << c.to_string() << std::endl << std::endl;
        c.update_value_at_idx(2.0f, 2);
        std::cout <<"Tensor c:" << std::endl << c.to_string() << std::endl << std::endl;
        Tensor<float> result = a.matmul(c);
        std::cout <<"Tensor cres:" << std::endl << result.to_string() << std::endl << std::endl;
        std::cout << "Performed matrix multiplication: " << a.shape_to_string() << " x " << c.shape_to_string() << " = " << result.shape_to_string() << std::endl;

        
        // Test transpose
        Tensor<float> transposed = result.transpose();
        std::cout << "Performed transpose: " << result.shape_to_string() << " -> " << transposed.shape_to_string() << std::endl;
        
        // Test softmax
        Tensor<float> d({2, 5});
        d.random_uniform(-2.0f, 2.0f);
        Tensor<float> softmax_result = d.softmax_axis1();
        std::cout << "Performed softmax on tensor: " << d.shape_to_string() << std::endl;
        
        std::cout << "All CUDA tensor tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#endif
