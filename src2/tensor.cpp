#include "neural_network.h"
#include <random>
#include <stdexcept>
#include <numeric>

Tensor::Tensor(const std::vector<size_t>& shape) : m_shape(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                      1ULL, std::multiplies<size_t>());
    m_data.resize(total_size);
}

void Tensor::zeros() {
    std::fill(m_data.begin(), m_data.end(), 0.0f);
}

void Tensor::random_init(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (auto& val : m_data) {
        val = dis(gen);
    }
}

Tensor Tensor::add(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::runtime_error("Tensor shapes don't match for addition");
    }
    
    Tensor result(m_shape);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::runtime_error("Tensor shapes don't match for element-wise multiplication");
    }
    
    Tensor result(m_shape);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] * other.m_data[i];
    }
    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    // Check if both tensors are 2D
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    
    // Check if inner dimensions match (MxN * NxP)
    if (m_shape[1] != other.m_shape[0]) {
        throw std::runtime_error(
            "Inner dimensions must match for matrix multiplication. Got shapes (" + 
            std::to_string(m_shape[0]) + "," + std::to_string(m_shape[1]) + ") and (" +
            std::to_string(other.m_shape[0]) + "," + std::to_string(other.m_shape[1]) + ")"
        );
    }
    
    std::vector<size_t> result_shape = {m_shape[0], other.m_shape[1]};
    Tensor result(result_shape);
    
    for (size_t i = 0; i < m_shape[0]; ++i) {
        for (size_t j = 0; j < other.m_shape[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < m_shape[1]; ++k) {
                sum += m_data[i * m_shape[1] + k] * 
                       other.m_data[k * other.m_shape[1] + j];
            }
            result.m_data[i * other.m_shape[1] + j] = sum;
        }
    }
    return result;
}

std::vector<float>& Tensor::data() {
    return m_data;
}

const std::vector<float>& Tensor::data() const {
    return m_data;
}

std::vector<size_t> Tensor::shape() const {
    return m_shape;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {  // Self-assignment check
        m_shape = other.m_shape;
        m_data = other.m_data;
    }
    return *this;
} 

Tensor Tensor::transpose() const {
    if (m_shape.size() != 2) {
        throw std::runtime_error("Transpose operation requires 2D tensor");
    }
    
    std::vector<size_t> transposed_shape = {m_shape[1], m_shape[0]};
    Tensor result(transposed_shape);
    
    for (size_t i = 0; i < m_shape[0]; ++i) {
        for (size_t j = 0; j < m_shape[1]; ++j) {
            result.m_data[j * m_shape[0] + i] = m_data[i * m_shape[1] + j];
        }
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                    1ULL, std::multiplies<size_t>());
    if (new_size != m_data.size()) {
        throw std::runtime_error("New shape must have same total size as original");
    }
    
    Tensor result(new_shape);
    result.m_data = m_data;  // Copy the data as-is
    return result;
} 