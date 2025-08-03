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


// TODOs
// - Implement the sum method
// - Implement the max method
// - Implement the exp method
// - Implement the log method
// - Implement the argmax method

// - on every method that creates a new tensor, check if the device is the same.
// - implement the kernels for cuda

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
        allocate_memory();
    }

    // Free the allocated memory
    void free_memory()
    {
        if (owns_data && data_ptr != nullptr)
        {
            delete[] data_ptr;
            data_ptr = nullptr;
        }
        owns_data = false;
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
            // TODO perfomance improvement: check if the shape is the same, and dont reallocate
            if (this->owns_data && this->shape != other.shape)
            {
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
    Tensor(const Tensor &other) : shape(other.shape), data_size(other.data_size), owns_data(other.owns_data)
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
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[i];
    }

    T &operator[](int i, int j)
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 2D tensor");
        }
        if (i < 0 || i >= shape[0] || j < 0 || j >= shape[1])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[i * shape[1] + j];
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

    // implement the *= operator for the tensor
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

    //// DEVICE MANAGEMENT ////

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

    // print the values of the tensor as a string up to 4 dims like a numpy array would be displayed with indicating the shape
    std::string to_string() const
    {
        std::stringstream ss;
        if (shape.size() == 1)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << data_ptr[i] << ", ";
            }
            ss << "]";
        }
        else if (shape.size() == 2)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << "[";
                for (int j = 0; j < shape[1]; j++)
                {
                    ss << data_ptr[i * shape[1] + j] << ", ";
                }
                ss << "], \n";
            }
            ss << "]";
        }
        else if (shape.size() == 3)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << "[";
                for (int j = 0; j < shape[1]; j++)
                {
                    ss << "[";
                    for (int k = 0; k < shape[2]; k++)
                    {
                        ss << data_ptr[i * shape[1] * shape[2] + j * shape[2] + k] << ", ";
                    }
                    ss << "], \n";
                }
                ss << "], \n";
            }
            ss << "]";
        }
        else if (shape.size() == 4)
        {
            ss << "[";
            for (int i = 0; i < shape[0]; i++)
            {
                ss << "[";
                for (int j = 0; j < shape[1]; j++)
                {
                    ss << "[";
                    for (int k = 0; k < shape[2]; k++)
                    {
                        ss << "[";
                        for (int l = 0; l < shape[3]; l++)
                        {
                            ss << data_ptr[i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l] << ", ";
                        }
                        ss << "], \n";
                    }
                    ss << "], \n";
                }
                ss << "], \n";
            }
            ss << "]";
        }
        else
        {
            throw std::invalid_argument("Tried to print tensor with more than 4 dimensions");
        }

        return ss.str();
    }

    // print the shape of the tensor
    std::string shape_to_string() const
    {
        std::stringstream ss;
        ss << "( ";
        for (int i = 0; i < shape.size() - 1; i++)
        {
            ss << shape[i] << ", ";
        }
        ss << shape[shape.size() - 1] << " )";
        return ss.str();
    }

    // Add method to compute the maximum along a specified axis
    Tensor<T> max(int axis, bool keepdims) const
    {
        // Implementation for max
        // This is a placeholder implementation
        // You need to implement the logic to compute the maximum along the specified axis
        return *this; // Return a new tensor with the maximum values
    }

    // Add method to compute the exponential of each element
    Tensor<T> exp() const
    {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size; i++)
        {
            result.data_ptr[i] = std::exp(data_ptr[i]);
        }
        return result;
    }

    // Add method to compute the sum along a specified axis
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
            }
        }
        else {
            throw std::invalid_argument("Sum not implemented for tensors with more than 2 dimensions");
        }
        return Tensor<T>(); // Return an empty tensor if the axis is not valid or the shape is not supported
    }

    // Add method to compute the natural logarithm of each element
    Tensor<T> log() const
    {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size; i++)
        {
            result.data_ptr[i] = std::log(data_ptr[i]);
        }
        return result;
    }

    // Add method to compute the index of the maximum value along a specified axis
    Tensor<int> argmax(int axis) const
    {
        // Implementation for argmax
        // This is a placeholder implementation
        // You need to implement the logic to find the index of the maximum value along the specified axis
        return Tensor<int>({1}); // Return a new tensor with the indices
    }

    // Add method to gather elements along a specified axis
    Tensor<T> gather(int axis, const Tensor<int> &indices) const
    {
        // Implementation for gather
        // This is a placeholder implementation
        // You need to implement the logic to gather elements along the specified axis
        return *this; // Return a new tensor with the gathered elements
    }



    ///// LEGAGCY METHODS /////

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

    T &operator[](int i, int j, int k)
    {
        if (shape.size() != 3)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 3D tensor");
        }
        if (i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || k < 0 || k >= shape[2])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[i * shape[1] * shape[2] + j * shape[2] + k];
    }

    T &operator[](int i, int j, int k, int l)
    {
        if (shape.size() != 4)
        {
            throw std::invalid_argument("Tried to access tensor with shape " + std::to_string(shape.size()) + " as 4D tensor");
        }
        if (i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || k < 0 || k >= shape[2] || l < 0 || l >= shape[3])
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data_ptr[i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l];
    }
};

#endif
