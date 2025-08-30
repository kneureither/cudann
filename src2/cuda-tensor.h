#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <memory>

class CUDATensor {
public:
    // Constructors and Destructor
    CUDATensor();
    CUDATensor(const std::vector<size_t>& shape);
    ~CUDATensor();

    // Copy and Move semantics
    CUDATensor(const CUDATensor& other);
    CUDATensor& operator=(const CUDATensor& other);
    CUDATensor(CUDATensor&& other) noexcept;
    CUDATensor& operator=(CUDATensor&& other) noexcept;

    // Memory Management
    void allocate();
    void free();

    // Data Transfer Methods
    void hostToDevice();
    void deviceToHost();

    // CUDA Kernel Wrappers for Basic Operations
    static CUDATensor add(const CUDATensor& a, const CUDATensor& b);
    static CUDATensor multiply(const CUDATensor& a, const CUDATensor& b);
    static CUDATensor dot(const CUDATensor& a, const CUDATensor& b);

    // Initialization Methods
    void zeros();
    void ones();
    void randomInit(float min = -1.0f, float max = 1.0f);

    // Accessor Methods
    float* data();
    const float* data() const;
    float* deviceData();
    const float* deviceData() const;

    std::vector<size_t> shape() const { return m_shape; }
    size_t totalElements() const;

private:
    // Host and Device Data Pointers
    float* m_hostData;
    float* m_deviceData;
    
    // Tensor Metadata
    std::vector<size_t> m_shape;
    size_t m_totalElements;

    // CUDA Error Checking
    void checkCudaError(cudaError_t err, const char* msg);
};

// CUDA Kernel Declarations
__global__ void addKernel(float* c, const float* a, const float* b, int n);
__global__ void multiplyKernel(float* c, const float* a, const float* b, int n);
__global__ void dotProductKernel(float* c, const float* a, const float* b, int m, int n, int k);
__global__ void zeroInitKernel(float* data, int n);
__global__ void randomInitKernel(float* data, int n, float min, float max, unsigned long long seed);
