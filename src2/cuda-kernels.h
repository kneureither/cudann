#include "cuda_tensor.h"
#include <curand.h>
#include <curand_kernel.h>

// Kernel for element-wise addition
__global__ void addKernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel for element-wise multiplication
__global__ void multiplyKernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Optimized Matrix Multiplication Kernel
__global__ void dotProductKernel(float* c, const float* a, const float* b, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Kernel to initialize all elements to zero
__global__ void zeroInitKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

// Kernel for random initialization
__global__ void randomInitKernel(float* data, int n, float min, float max, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        data[idx] = min + (max - min) * curand_uniform(&state);
    }
}

// CUDATensor Implementation
CUDATensor::CUDATensor(const std::vector<size_t>& shape) 
    : m_shape(shape), m_hostData(nullptr), m_deviceData(nullptr) {
    m_totalElements = 1;
    for (auto dim : shape) {
        m_totalElements *= dim;
    }
    allocate();
}

CUDATensor CUDATensor::add(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }

    CUDATensor result(a.shape());
    int blockSize = 256;
    int numBlocks = (a.totalElements() + blockSize - 1) / blockSize;

    addKernel<<<numBlocks, blockSize>>>(
        result.deviceData(), 
        a.deviceData(), 
        b.deviceData(), 
        a.totalElements()
    );

    result.deviceToHost();
    return result;
}

// Similar implementations for multiply and dot product kernels...

void CUDATensor::randomInit(float min, float max) {
    int blockSize = 256;
    int numBlocks = (m_totalElements + blockSize - 1) / blockSize;

    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    randomInitKernel<<<numBlocks, blockSize>>>(
        m_deviceData, 
        m_totalElements, 
        min, 
        max, 
        seed
    );

    deviceToHost();
}
