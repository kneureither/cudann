#pragma once

#include "cuda_tensor.h"

class CUDALayer {
public:
    // Virtual forward and backward propagation
    virtual CUDATensor forward(const CUDATensor& input) = 0;
    virtual CUDATensor backward(const CUDATensor& outputGradient, float learningRate) = 0;

protected:
    // Common layer components
    CUDATensor weights;
    CUDATensor biases;
};

class CUDADenseLayer : public CUDALayer {
public:
    CUDADenseLayer(size_t inputSize, size_t outputSize);

    CUDATensor forward(const CUDATensor& input) override;
    CUDATensor backward(const CUDATensor& outputGradient, float learningRate) override;

private:
    // Activation function (can be a function pointer or strategy)
    CUDATensor activationForward(const CUDATensor& input);
    CUDATensor activationBackward(const CUDATensor& gradient);

    // Last input for backpropagation
    CUDATensor m_lastInput;
};

// CUDA Activation Function Kernels
__global__ void reluForwardKernel(float* output, const float* input, int n);
__global__ void reluBackwardKernel(float* gradient, const float* input, int n);
