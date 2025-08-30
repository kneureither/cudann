#include "neural_network.h"
#include <stdexcept>
#include <iostream>

// ReLU Implementation
float ReLU::activate(float x) const
{
    return std::max(0.0f, x);
}

float ReLU::derivative(float x) const
{
    return x > 0.0f ? 1.0f : 0.0f;
}

// Sigmoid Implementation
float Sigmoid::activate(float x) const
{
    return 1.0f / (1.0f + std::exp(-x));
}

float Sigmoid::derivative(float x) const
{
    float sigmoid_x = activate(x);
    return sigmoid_x * (1.0f - sigmoid_x);
}

// Layer Implementation
Layer::Layer(size_t input_size, size_t output_size)
    : m_input_size(input_size), m_output_size(output_size),
      weights({input_size, output_size}),
      biases({1, output_size})
{
    weights.random_init();
    biases.zeros();
}

// Dense Layer Implementation
DenseLayer::DenseLayer(size_t input_size,
                       size_t output_size,
                       std::unique_ptr<ActivationFunction> act_func)
    : Layer(input_size, output_size)
{
    activation = std::move(act_func);
}

Tensor DenseLayer::forward(const Tensor &input)
{
    m_last_input = input;
    Tensor output = input.dot(weights);

    // Add biases to each row
    for (size_t i = 0; i < output.shape()[0]; ++i)
    {
        for (size_t j = 0; j < output.shape()[1]; ++j)
        {
            output.data()[i * output.shape()[1] + j] += biases.data()[j];
        }
    }

    // Apply activation function using vector operations
    std::vector<float> activated = activation->activate_vector(output.data());
    output.data() = activated;

    return output;
}

Tensor DenseLayer::backward(const Tensor &output_gradient, float learning_rate)
{
    // Calculate gradient with respect to activation
    Tensor activation_gradient = output_gradient;
    for (size_t i = 0; i < activation_gradient.data().size(); ++i)
    {
        std::vector<float> derivatives = activation->derivative_vector(activation_gradient.data());
        activation_gradient.data()[i] *= derivatives[i];
    }

    // We need to reshape activation_gradient from (1,10) to (10,1) for the dot product
    // Add this to your Tensor class:
    Tensor reshaped_activation = activation_gradient.reshape({activation_gradient.shape()[1], 1});
    
    // Now shapes will be: (1,784) dot (10,1) -> (784,10)
    Tensor weights_gradient = m_last_input.transpose().dot(activation_gradient);
    
    // For this: (1,10) dot (784,10)' -> (1,784)
    Tensor input_gradient = activation_gradient.dot(weights.transpose());

    // Update weights and biases
    for (size_t i = 0; i < weights.data().size(); ++i)
    {
        weights.data()[i] -= learning_rate * weights_gradient.data()[i];
    }

    for (size_t i = 0; i < biases.data().size(); ++i)
    {
        biases.data()[i] -= learning_rate * activation_gradient.data()[i];
    }

    return input_gradient;
}

// MSE Loss Implementation
float MSELoss::compute(const Tensor &predicted, const Tensor &target)
{
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.data().size(); ++i)
    {
        float diff = predicted.data()[i] - target.data()[i];
        sum += diff * diff;
    }
    return sum / predicted.data().size();
}

Tensor MSELoss::gradient(const Tensor &predicted, const Tensor &target)
{
    Tensor gradient(predicted.shape());
    for (size_t i = 0; i < predicted.data().size(); ++i)
    {
        gradient.data()[i] = 2.0f * (predicted.data()[i] - target.data()[i]) / predicted.data().size();
    }
    return gradient;
}

// Cross Entropy Loss Implementation
float CrossEntropyLoss::compute(const Tensor& predicted, const Tensor& target) {
    float loss = 0.0f;
    const size_t num_classes = predicted.shape()[1];
    
    for (size_t j = 0; j < num_classes; ++j) {
        // Add small epsilon to avoid log(0)
        const float epsilon = 1e-7f;
        float pred = std::max(std::min(predicted.data()[j], 1.0f - epsilon), epsilon);
        loss -= target.data()[j] * std::log(pred);
    }
    return loss;
}

Tensor CrossEntropyLoss::gradient(const Tensor& predicted, const Tensor& target) {
    Tensor gradient(predicted.shape());
    const size_t num_classes = predicted.shape()[1];
    
    for (size_t j = 0; j < num_classes; ++j) {
        // Add small epsilon to avoid division by zero
        const float epsilon = 1e-7f;
        float pred = std::max(std::min(predicted.data()[j], 1.0f - epsilon), epsilon);
        gradient.data()[j] = -(target.data()[j] / pred);
    }
    return gradient;
}

// Neural Network Implementation
void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer)
{
    m_layers.push_back(std::move(layer));
}

void NeuralNetwork::set_loss_function(std::unique_ptr<LossFunction> loss_func)
{
    m_loss_function = std::move(loss_func);
}

LossFunction& NeuralNetwork::NeuralNetwork::get_loss_function()
{
    if (!m_loss_function)
    {
        throw std::runtime_error("No loss function has been set");
    }
    return *m_loss_function;
}

Tensor NeuralNetwork::predict(const Tensor &input)
{
    Tensor current = input;
    for (const auto &layer : m_layers)
    {
        current = layer->forward(current);
    }
    return current;
}

void NeuralNetwork::train(const Tensor &input, const Tensor &target,
                          float learning_rate)
{
    // Forward pass
    Tensor output = predict(input);

    // Backward pass
    Tensor gradient = m_loss_function->gradient(output, target);
    //std::cout << "Loss Gradient done" << std::endl;
    int count = 0;
    //std::cout << "Backward pass, layers: " << m_layers.size() << std::endl;
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
        //std::cout << "layer Gradient " << count; 
        gradient = (*it)->backward(gradient, learning_rate);
        count++;
    }
}