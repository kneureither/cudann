#pragma once

#include "cuda_layer.h"
#include <vector>
#include <memory>

class CUDANeuralNetwork {
public:
    void addLayer(std::unique_ptr<CUDALayer> layer);
    CUDATensor predict(const CUDATensor& input);
    void train(const CUDATensor& input, const CUDATensor& target, float learningRate);

private:
    std::vector<std::unique_ptr<CUDALayer>> m_layers;
};
