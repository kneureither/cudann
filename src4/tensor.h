#ifndef TENSOR_H
#define TENSOR_H
#define CUDA_AVAILABLE 0

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <cmath>

#include "utils.h"


enum class Device
{
    CPU,
    GPU
};

template <typename T>
class Tensor
{
private:
    T *data_ptr;               // Pointer to the data array
    size_t data_size;             // Total number of elements in the tensor
    bool owns_data;            // Flag to track ownership for proper cleanup
    std::vector<T *> view_idx; // View indices for easier element access
    Device device;             // Device the tensor is on

    // Memory management methods
    void allocate_memory()
    {
        if (data_size > 0)
        {
            data_ptr = new T[data_size];
            owns_data = true;
        }
    }

    // Constructor for Tensor with specified shape
    void allocate_memory(std::vector<size_t> shape)
    {
        this->shape = shape;
        data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        logger("allocate_memory(shape) data size: " + std::to_string(data_size), "DEBUG", __FILE__, __LINE__);
        allocate_memory();
    }

    // Free the allocated memory
    void free_memory()
    {
        if (owns_data && data_ptr != nullptr)
        {
            logger("freeing memory", "DEBUG", __FILE__, __LINE__);
            delete[] data_ptr;
            data_ptr = nullptr;
        }
        owns_data = false;
        this->data_size = 0;
        this->shape = std::vector<size_t>({});
    }

#if CUDA_AVAILABLE
    // allocate memory on cuda device
    void allocate_cuda_memory()
    {
        cudaMalloc(&data_ptr, data_size * sizeof(T));
        owns_data = true;
    }

    // allocate memory on cuda device with specified shape
    void allocate_cuda_memory(std::vector<size_t> shape)
    {
        this->shape = shape;
        data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        allocate_cuda_memory();
    }

    // free memory on cuda device
    void free_cuda_memory()
    {
        if (owns_data && data_ptr != nullptr)
        {
            cudaFree(data_ptr);
            data_ptr = nullptr;
        }
    }

    // copy data from cpu to cuda device
    void copy_to_cuda()
    {
        cudaMemcpy(data_ptr, data_ptr, data_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // copy data from cuda device to cpu
    void copy_to_cpu()
    {
        cudaMemcpy(data_ptr, data_ptr, data_size * sizeof(T), cudaMemcpyDeviceToHost);
    }
#endif
public:
    std::vector<size_t> shape;
    // Default constructor
    Tensor() : data_ptr(nullptr), data_size(0), owns_data(false) {}

    // Constructor for Tensor with specified shape
    Tensor(std::vector<size_t> shape) : shape(shape), data_size(0), owns_data(false)
    {
        allocate_memory(shape);
    }

    Tensor(std::vector<size_t> shape, Device device) : shape(shape), data_size(0), owns_data(false), device(device)
    {
        allocate_memory(shape);
    }

    // Assignment operator
    Tensor<T> &operator=(const Tensor<T> &other)
    {
        if (this != &other)
        {
            logger("assigning new tensor, old shape " + this->shape_to_string() + " other shape: " + other.shape_to_string() +
                       " this owns data " + std::to_string(this->owns_data),
                   "DEBUG", __FILE__, __LINE__);
            // TODO perfomance improvement: check if the shape is the same, and dont reallocate
            if (this->owns_data)
            {
                logger("freeing memory", "DEBUG", __FILE__, __LINE__);
                free_memory();
            }
            shape = other.shape;
            data_size = other.data_size;
            device = other.device;
            allocate_memory(shape);
            for (int i = 0; i < data_size; i++)
            {
                data_ptr[i] = other.data_ptr[i];
            }
            owns_data = true;
        }
        return *this;
    }

    // Copy constructor
    Tensor(const Tensor &other) : shape(other.shape), data_size(other.data_size), owns_data(false)
    {
        allocate_memory(other.shape);
        for (int i = 0; i < data_size; i++)
        {
            data_ptr[i] = other.data_ptr[i];
        }
    }
    void zeros()
    {
        for (int i = 0; i < data_size; i++) data_ptr[i] = T();
    }

    // Set the tensor to ones
    void ones()
    {
        for (int i = 0; i < data_size; i++) data_ptr[i] = T(1);
    }

    // Randomize the tensor
    void random()
    {
        for (int i = 0; i < data_size; i++) data_ptr[i] = T(static_cast<double>(rand()) / RAND_MAX);
    }

    // Randomize the tensor with a glorot initialization
    void random_init()
    {
        for (int i = 0; i < data_size; i++) data_ptr[i] = T(rand() % RAND_MAX);
    }

    void random_uniform(T min, T max)
    {
        for (int i = 0; i < data_size; i++)
        {
            data_ptr[i] = min + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max - min)));
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

    // access the tensor as a flat array
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
        return data_ptr[i];
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
        return data_ptr[i];
    }

    // 2D indexer (operator())
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
        return data_ptr[static_cast<size_t>(i) * shape[1] + static_cast<size_t>(j)];
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
        return data_ptr[static_cast<size_t>(i) * shape[1] + static_cast<size_t>(j)];
    }

    //// OTHER OPERATORS ////

    // implement the += operator for the tensor
    Tensor<T> &operator+=(const Tensor<T> &other)
    {
        if (shape == other.shape)
        {
            logger("Adding tensors with shape: " + shape_to_string() + " and " + other.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            for (int i = 0; i < data_size; i++)
            {
                data_ptr[i] += other.data_ptr[i];
            }
            return *this;
        } 
        // adding a vector to matrix (e.g. [batch, features] + [features])
        else if (shape.size() == 2 && other.shape.size() == 1 && shape[1] == other.shape[0])
        {
            logger("Adding matrix with shape: " + shape_to_string() + " and vector with shape: " + other.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    //logger("Adding i,j (" + std::to_string(i) + "," + std::to_string(j) + ") " + std::to_string(data_ptr[i * shape[1] + j]) + " and " + std::to_string(other.data_ptr[j]), "DEBUG", __FILE__, __LINE__);
                    data_ptr[i * shape[1] + j] += other.data_ptr[j];
                }
            }
            return *this;
        }
        // adding a scalar to the tensor
        else if (other.shape.size() == 1 && other.shape[0] == 1 && shape.size() > 0)
        {   
            logger("Adding scalar to tensor with shape: " + shape_to_string() + " and scalar: " + std::to_string(other.data_ptr[0]), "DEBUG", __FILE__, __LINE__);
            for (int i = 0; i < data_size; i++)
            {
                data_ptr[i] += other.data_ptr[0];
            }
            return *this;
        }
        else
        {
            throw std::invalid_argument("Tried to add tensors with different shapes: " + shape_to_string() + " and " + other.shape_to_string());
        }
    }

    // implement the -= operator for the tensor
    Tensor<T> &operator-=(const Tensor<T> &other)
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tried to subtract tensors with different shapes");
        }
        for (int i = 0; i < data_size; i++)
        {
            data_ptr[i] -= other.data_ptr[i];
        }
        return *this;
    }

    // implement the *= as element-wise * operator for the tensor
    Tensor<T> &operator*=(const Tensor<T> &other)
    {
        if (shape != other.shape) throw std::invalid_argument("Tried to multiply tensors with different shapes");
        for (int i = 0; i < data_size; i++) data_ptr[i] *= other.data_ptr[i];
        return *this;
    }

    // implement the '*=' operator for multiplying a tensor by a scalar
    Tensor<T> &operator*=(const T &scalar)
    {
        for (int i = 0; i < data_size; i++) data_ptr[i] *= scalar;
        return *this;
    }

    // implement the '>' operator for masking
    Tensor<T> operator>(const T &scalar) const {
        Tensor<T> result(shape);
        for(int i=0; i<data_size; i++) {
            result.data_ptr[i] = static_cast<T> (data_ptr[i] > scalar);
        }
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

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < other.shape[1]; j++)
            {
                T sum = 0;
                for (int k = 0; k < shape[1]; k++)
                {
                    sum += data_ptr[i * shape[1] + k] * other.data_ptr[k * other.shape[1] + j];
                }
                result.data_ptr[i * other.shape[1] + j] = sum;
            }
        }

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
            // 2D matrix transpose: swap dimensions
            Tensor<T> result({shape[1], shape[0]});
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    result.data_ptr[j * shape[0] + i] = data_ptr[i * shape[1] + j];
                }
            }
            return result;
        }
        else
        {
            throw std::invalid_argument("Transpose not implemented for tensors with more than 2 dimensions");
        }
    }

    // method to compute the exponential of each element
    Tensor<T> exp() const
    {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size; i++)
        {
            result.data_ptr[i] = std::exp(this->data_ptr[i]);
        }
        return result;
    }

    // method to compute the natural logarithm of each element
    Tensor<T> log() const
    {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size; i++)
        {
            result.data_ptr[i] = std::log(data_ptr[i]);
        }
        return result;
    }

    // method to compute the sum along a specified axis
    Tensor<T> sum(int axis) const
    {
        //check if axis is valid
        if (axis < 0 || axis >= shape.size())
        {
            throw std::invalid_argument("Axis out of bounds");
        }
        if (shape.size() == 1)
        {
            // sum along the only axis
            T sum = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                sum += data_ptr[i];
            }
            Tensor<T> result({1});
            result[0] = sum;
            return result;
        }
        else if (shape.size() == 2)
        {
            // sum along the specified axis
            if (axis == 0)
            {
                // sum along rows
                Tensor<T> result({shape[1]});
                for (int j = 0; j < shape[1]; j++)
                {
                    T sum = 0;
                    for (int i = 0; i < shape[0]; i++)
                    {
                        sum += data_ptr[i * shape[1] + j];
                    }
                    result.data_ptr[j] = sum;
                }
                return result;
            }
            else if (axis == 1)
            {
                // sum along columns
                Tensor<T> result({shape[0]});
                for (int i = 0; i < shape[0]; i++)
                {
                    T sum = 0;
                    for (int j = 0; j < shape[1]; j++)
                    {
                        sum += data_ptr[i * shape[1] + j];
                    }
                    result.data_ptr[i] = sum;
                }
                return result;
            } else {
                throw std::invalid_argument("sum(): axis must be 0 or 1, was " + std::to_string(axis));
            }
        }
        else {
            throw std::invalid_argument("Sum not implemented for tensors with more than 2 dimensions");
        }
        return Tensor<T>(); // Return an empty tensor if the axis is not valid or the shape is not supported
    }

    // method to compute the argmax along a specified axis
    Tensor<int> argmax(int axis) const
    {
        if (shape.size() == 1) {
            Tensor<int> out({1});
            const size_t B = shape[0];
            size_t best_i = 0;
            T best_v = data_ptr[0];
            for (size_t i = 1; i < B; i++) {
                T v = data_ptr[i];
                if (v > best_v)
                {
                    best_v = v;
                    best_i = i;
                }
            }
            out.get_data_ptr()[0] = static_cast<int>(best_i);
            return out;
        } else if (shape.size() == 2) {
            const size_t B = shape[0], C = shape[1];
            if (axis == 0)
            { // along rows → [C]
                Tensor<int> out({C});
                for (size_t j = 0; j < C; ++j)
                {
                    size_t best_i = 0;
                    T best_v = data_ptr[0 * C + j];
                    for (size_t i = 1; i < B; ++i)
                    {
                        T v = data_ptr[i * C + j];
                        if (v > best_v)
                        {
                            best_v = v;
                            best_i = i;
                        }
                    }
                    out.get_data_ptr()[j] = static_cast<int>(best_i);
                }
                return out;
            }
            else if (axis == 1)
            { // along cols → [B]
                Tensor<int> out({B});
                for (size_t i = 0; i < B; ++i)
                {
                    size_t best_j = 0;
                    T best_v = data_ptr[i * C + 0];
                    for (size_t j = 1; j < C; ++j)
                    {
                        T v = data_ptr[i * C + j];
                        if (v > best_v)
                        {
                            best_v = v;
                            best_j = j;
                        }
                    }
                    out.get_data_ptr()[i] = static_cast<int>(best_j);
                }
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

    // Softmax over axis=1 (rows). Stable: subtract row max, then exp/sum.
    Tensor<T> softmax_axis1() const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("softmax_axis1 expects a 2D tensor");
        }
        const size_t B = shape[0], C = shape[1];

        Tensor<T> out({B, C});
        for (size_t i = 0; i < B; ++i)
        {
            // 1) row max
            T m = data_ptr[i * C + 0];
            for (size_t j = 1; j < C; ++j)
            {
                T v = data_ptr[i * C + j];
                if (v > m)
                    m = v;
            }
            // 2) sum exp(z - m)
            T sum_exp = T(0);
            for (size_t j = 0; j < C; ++j)
            {
                T e = std::exp(data_ptr[i * C + j] - m);
                out.get_data_ptr()[i * C + j] = e; // temp store e
                sum_exp += e;
            }
            // 3) normalize
            for (size_t j = 0; j < C; ++j)
            {
                out.get_data_ptr()[i * C + j] /= sum_exp;
            }
        }
        return out;
    }

    // Log-softmax over axis=1 (rows). Stable version.
    Tensor<T> log_softmax_axis1() const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("log_softmax_axis1 expects a 2D tensor");
        }
        const size_t B = shape[0], C = shape[1];

        Tensor<T> out({B, C});
        for (size_t i = 0; i < B; ++i)
        {
            // 1) row max
            T m = data_ptr[i * C + 0];
            for (size_t j = 1; j < C; ++j)
            {
                T v = data_ptr[i * C + j];
                if (v > m)
                    m = v;
            }
            // 2) logsumexp = m + log(sum exp(z - m))
            T sum_exp = T(0);
            for (size_t j = 0; j < C; ++j)
            {
                sum_exp += std::exp(data_ptr[i * C + j] - m);
            }
            T lse = m + std::log(sum_exp);
            // 3) logp = z - lse
            for (size_t j = 0; j < C; ++j)
            {
                out.get_data_ptr()[i * C + j] = data_ptr[i * C + j] - lse;
            }
        }
        return out;
    }

    // Add this method implementation for CPU tensors
    void scatter_subtract_axis1(const Tensor<int>& indices, T value) {
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
        
        // CPU implementation - simple loop
        for (size_t row = 0; row < rows; ++row) {
            int col_idx = indices[row];
            if (col_idx >= 0 && col_idx < static_cast<int>(cols)) {
                (*this)(row, col_idx) -= value;
            }
        }
    }


    //// DEVICE MANAGEMENT AND UTILS ////

    // bring the tensor to a specific device
    void to_device(Device device)
    {
        if (device == this->device)
        {
            // do nothing
        }
        else if (device == Device::CPU && this->device == Device::GPU)
        {
            throw std::runtime_error("Tried to move tensor from GPU to CPU");
            // move the tensor to the CPU
            T *cpu_data = new T[data_size];
            for (int i = 0; i < data_size; i++)
            {
                cpu_data[i] = data_ptr[i];
            }
        }
        else if (device == Device::GPU && this->device == Device::CPU)
        {
            throw std::runtime_error("Tried to move tensor from CPU to GPU");
            // move the tensor to the GPU
            T *gpu_data = new T[data_size];
            for (int i = 0; i < data_size; i++)
            {
                gpu_data[i] = data_ptr[i];
            }
        }
        else
        {
            throw std::invalid_argument("Tried to move tensor to the same device");
        }
    }

    // get the device the tensor is on
    Device get_device() const { return device; }

    std::string to_string() const
    {
        std::stringstream ss;
        if (shape.size() == 1)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << data_ptr[i];
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
                    ss << data_ptr[i * shape[1] + j];
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

    // print the shape of the tensor
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

    /*
    ///// LEGAGCY METHODS /////

        Tensor<T> reduce_max(int axis, bool keepdim) const
    {
        if (shape.size() != 2)
            throw std::invalid_argument("reduce_max: 2D only");
        const size_t B = shape[0], C = shape[1];

        if (axis == 0)
        { // max over rows → [C] or [1,C]
            Tensor<T> out(keepdim ? std::vector<size_t>{1, C} : std::vector<size_t>{C});
            for (size_t j = 0; j < C; ++j)
            {
                T m = data_ptr[0 * C + j];
                for (size_t i = 1; i < B; ++i)
                {
                    T v = data_ptr[i * C + j];
                    if (v > m)
                        m = v;
                }
                if (keepdim)
                    out.get_data_ptr()[j] = m;
                else
                    out.get_data_ptr()[j] = m;
            }
            return out;
        }
        else if (axis == 1)
        { // max over cols → [B] or [B,1]
            Tensor<T> out(keepdim ? std::vector<size_t>{B, 1} : std::vector<size_t>{B});
            for (size_t i = 0; i < B; ++i)
            {
                T m = data_ptr[i * C + 0];
                for (size_t j = 1; j < C; ++j)
                {
                    T v = data_ptr[i * C + j];
                    if (v > m)
                        m = v;
                }
                out.get_data_ptr()[i] = m; // layout works for both shapes here
            }
            return out;
        }
        else
        {
            throw std::invalid_argument("reduce_max: axis must be 0 or 1");
        }
    }

    Tensor<T> dot_1d(const Tensor<T> &other) const
    {
        if (shape.size() != 1 || other.shape.size() != 1)
        {
            throw std::invalid_argument("Tried to dot product tensors with shape " + std::to_string(shape.size()) + " and " + std::to_string(other.shape.size()));
        }
        if (shape[0] != other.shape[0])
        {
            throw std::invalid_argument("Tried to dot product tensors with shape " + std::to_string(shape.size()) + " and " + std::to_string(other.shape.size()));
        }

        Tensor<T> result({1});
        for (int i = 0; i < shape[0]; i++)
        {
            result[0] += data_ptr[i] * other.data_ptr[i];
        }
        return result;
    }

    Tensor<T> dot_2d(const Tensor<T> &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 1)
        {
            throw std::invalid_argument("Tried to dot product tensors with shape " + std::to_string(shape.size()) + " and " + std::to_string(other.shape.size()));
        }
        if (shape[1] != other.shape[0])
        {
            throw std::invalid_argument("Tried to dot product tensors with shape " + shape_to_string() + " and " + other.shape_to_string());
        }

        Tensor<T> result({shape[0], other.shape[1]});
        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < other.shape[1]; j++)
            {
                T sum = 0;
                for (int k = 0; k < shape[1]; k++)
                {
                    sum += data_ptr[i * shape[1] + k] * other.data_ptr[k * other.shape[1] + j];
                }
                result.data_ptr[i * other.shape[1] + j] = sum;
            }
        }

        return result;
    }

    // 3D indexer (operator())
    T &operator()(int i, int j, int k)
    {
        if (shape.size() != 3)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 3D tensor");
        }
        if (i < 0 || j < 0 || k < 0 ||
            static_cast<size_t>(i) >= shape[0] ||
            static_cast<size_t>(j) >= shape[1] ||
            static_cast<size_t>(k) >= shape[2])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[static_cast<size_t>(i) * shape[1] * shape[2] +
                        static_cast<size_t>(j) * shape[2] +
                        static_cast<size_t>(k)];
    }
    const T &operator()(int i, int j, int k) const
    {
        if (shape.size() != 3)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 3D tensor");
        }
        if (i < 0 || j < 0 || k < 0 ||
            static_cast<size_t>(i) >= shape[0] ||
            static_cast<size_t>(j) >= shape[1] ||
            static_cast<size_t>(k) >= shape[2])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[static_cast<size_t>(i) * shape[1] * shape[2] +
                        static_cast<size_t>(j) * shape[2] +
                        static_cast<size_t>(k)];
    }

    // 4D indexer (operator())
    T &operator()(int i, int j, int k, int l)
    {
        if (shape.size() != 4)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 4D tensor");
        }
        if (i < 0 || j < 0 || k < 0 || l < 0 ||
            static_cast<size_t>(i) >= shape[0] ||
            static_cast<size_t>(j) >= shape[1] ||
            static_cast<size_t>(k) >= shape[2] ||
            static_cast<size_t>(l) >= shape[3])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[static_cast<size_t>(i) * shape[1] * shape[2] * shape[3] +
                        static_cast<size_t>(j) * shape[2] * shape[3] +
                        static_cast<size_t>(k) * shape[3] +
                        static_cast<size_t>(l)];
    }
    const T &operator()(int i, int j, int k, int l) const
    {
        if (shape.size() != 4)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 4D tensor");
        }
        if (i < 0 || j < 0 || k < 0 || l < 0 ||
            static_cast<size_t>(i) >= shape[0] ||
            static_cast<size_t>(j) >= shape[1] ||
            static_cast<size_t>(k) >= shape[2] ||
            static_cast<size_t>(l) >= shape[3])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[static_cast<size_t>(i) * shape[1] * shape[2] * shape[3] +
                        static_cast<size_t>(j) * shape[2] * shape[3] +
                        static_cast<size_t>(k) * shape[3] +
                        static_cast<size_t>(l)];
    }
    */
};

#endif
