#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Forward declarations
class Layer;
class Tensor;

// Base Tensor class - will be crucial for CUDA implementation later
class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<size_t>& shape);
    
    // Core tensor operations
    virtual void zeros();
    virtual void random_init(float min = -1.0f, float max = 1.0f);
    
    // Basic mathematical operations
    virtual Tensor add(const Tensor& other) const;
    virtual Tensor multiply(const Tensor& other) const;
    virtual Tensor dot(const Tensor& other) const;

    // Getters and setters
    std::vector<float>& data();
    const std::vector<float>& data() const;
    std::vector<size_t> shape() const;
    Tensor &operator=(const Tensor &other); // Assignment operator

    Tensor transpose() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;

private:
    std::vector<float> m_data;
    std::vector<size_t> m_shape;
};

// Activation function interface with vector-based operations
class ActivationFunction {
public:
    // Single value operations (mainly for non-softmax activations)
    virtual float activate(float x) const = 0;
    virtual float derivative(float x) const = 0;
    
    // Vector operations (needed for softmax and can be used by others)
    virtual std::vector<float> activate_vector(const std::vector<float>& x) const {
        // Default implementation for most activation functions
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = activate(x[i]);
        }
        return result;
    }
    
    virtual std::vector<float> derivative_vector(const std::vector<float>& x) const {
        // Default implementation for most activation functions
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = derivative(x[i]);
        }
        return result;
    }
    
    virtual ~ActivationFunction() = default;
};

// Softmax activation function
class Softmax : public ActivationFunction {
public:
    float activate(float x) const override {
        // Not meaningful for single value
        throw std::runtime_error("Softmax cannot be computed on a single value");
    }

    float derivative(float x) const override {
        // Not meaningful for single value
        throw std::runtime_error("Softmax derivative cannot be computed on a single value");
    }

    std::vector<float> activate_vector(const std::vector<float>& x) const override {
        std::vector<float> output(x.size());
        
        // Find max for numerical stability
        float max_val = *std::max_element(x.begin(), x.end());
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = std::exp(x[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] /= sum;
        }
        
        return output;
    }

    std::vector<float> derivative_vector(const std::vector<float>& x) const override {
        // For softmax, the Jacobian is a matrix, but we typically use this
        // in combination with cross-entropy loss where it simplifies
        // This implementation assumes it will be used with cross-entropy loss
        return activate_vector(x);
    }
};



// Concrete activation functions
class ReLU : public ActivationFunction {
public:
    float activate(float x) const override;
    float derivative(float x) const override;
};

class Sigmoid : public ActivationFunction {
public:
    float activate(float x) const override;
    float derivative(float x) const override;
};

// Base Layer class
class Layer {
public:
    Layer(size_t input_size, size_t output_size);
    
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient, float learning_rate) = 0;

    // Common layer properties
    Tensor weights;
    Tensor biases;
    std::unique_ptr<ActivationFunction> activation;

protected:
    size_t m_input_size;
    size_t m_output_size;
};

// Dense (Fully Connected) Layer
class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, 
               size_t output_size, 
               std::unique_ptr<ActivationFunction> act_func);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, float learning_rate) override;

private:
    Tensor m_last_input;
};

// Loss Function Interface
class LossFunction {
public:
    virtual float compute(const Tensor& predicted, const Tensor& target) = 0;
    virtual Tensor gradient(const Tensor& predicted, const Tensor& target) = 0;
    virtual ~LossFunction() = default;
};

// Mean Squared Error Loss
class MSELoss : public LossFunction {
public:
    float compute(const Tensor& predicted, const Tensor& target) override;
    Tensor gradient(const Tensor& predicted, const Tensor& target) override;
};

// Cross Entropy Loss
class CrossEntropyLoss : public LossFunction
{
public:
    float compute(const Tensor &predicted, const Tensor &target) override;
    Tensor gradient(const Tensor &predicted, const Tensor &target) override;
};

// Neural Network Model
class NeuralNetwork {
public:
    void add_layer(std::unique_ptr<Layer> layer);
    Tensor predict(const Tensor& input);
    void train(const Tensor& input, const Tensor& target, float learning_rate);
    void set_loss_function(std::unique_ptr<LossFunction> loss_func);
    LossFunction& get_loss_function();

private:
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::unique_ptr<LossFunction> m_loss_function;
};

#endif // NEURAL_NETWORK_H