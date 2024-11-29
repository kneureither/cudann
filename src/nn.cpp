/******************************************************************************
 * Copyright (c) 2024
 *
 * Author: Konstantin Neureither
 * Date: 2024-11-25
 *****************************************************************************/

#include "nn.h"
#include <cstdlib>
#include <cmath>

NeuralNetwork::NeuralNetwork(int n_inputs, int n_outputs)
{
    this->n_inputs = n_inputs;
    this->n_outputs = n_outputs;
    w1 = new double[n_inputs * n_outputs];
    w1_grad = new double[n_inputs * n_outputs];
    b1 = new double[n_outputs];
    b1_grad = new double[n_outputs];
    output = new double[n_outputs];
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] w1;
    delete[] b1;
    delete[] w1_grad;
    delete[] b1_grad;
    delete[] output;
}

/*
    Randomizes weights and biases.
*/
void NeuralNetwork::randomize_weights()
{
    for (int i = 0; i < n_inputs * n_outputs; i++)
    {
        this->w1[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n_outputs; i++)
    {
        this->b1[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

double inline NeuralNetwork::activation_function(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/*
    Computes softmax of a vector of activations.
    Input: vector of activations (length = n_outputs)
    Output: vector of probabilities (length = n_outputs)
*/
void NeuralNetwork::neural_network_softmax(double *activations, int length)
{
    int i;
    float sum, max;

    // find max
    for (i = 1, max = activations[0]; i < length; i++)
    {
        if (activations[i] > max)
        {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++)
    {
        activations[i] = exp(activations[i] - max); // avoid overflow
        sum += activations[i];                      // sum of probabilities for denominator
    }

    for (i = 0; i < length; i++)
    {
        activations[i] /= sum;
    }
}

/*
    Perform single forward pass of one single sample
*/
void NeuralNetwork::forward(double *input)
{
    int i, j;
    double sum;

    // go over all outputs: j counts outputs
    for (j = 0; j < n_outputs; j++)
    {
        sum = 0.0;

        // sum over all inputs to one single output: i counts inputs
        for (i = 0; i < n_inputs; i++)
        {
            sum += input[i] * w1[j * n_inputs + i];
        }

        // add bias
        output[j] = sum + b1[j];
    }

    // apply softmax
    neural_network_softmax(output, n_outputs);
}

/*
    Computes loss for a single sample.
*/
double NeuralNetwork::loss_function(double *target)
{
    double loss = 0.0;
    for (int i = 0; i < n_outputs; i++)
    {
        loss += -target[i] * std::log(output[i]);
    }
    return loss;
}

void NeuralNetwork::backprop(double *input, double *target)
{
    int i, j;

    // For cross-entropy loss with softmax, the gradient simplifies to (output - target)
    // This is because the derivatives of softmax and cross-entropy cancel out nicely
    for (j = 0; j < n_outputs; j++)
    {
        // Compute error term (output - target)
        double error = output[j] - target[j];

        // Update bias gradient
        b1_grad[j] += error;

        // Update weight gradients for all inputs to this output
        for (i = 0; i < n_inputs; i++)
        {
            w1_grad[j * n_inputs + i] += error * input[i];
        }
    }
}

void NeuralNetwork::zero_gradients()
{
    int i, j;

    for (j = 0; j < n_outputs; j++)
    {
        b1_grad[j] = 0.0;
        for (i = 0; i < n_inputs; i++)
        {
            w1_grad[j * n_inputs + i] = 0.0;
        }
    }
}

void NeuralNetwork::update_weights_sgd(double learning_rate)
{
    int i, j;

    for (j = 0; j < n_outputs; j++)
    {
        b1[j] -= learning_rate * b1_grad[j];

        for (i = 0; i < n_inputs; i++)
        {
            w1[j * n_inputs + i] -= learning_rate * w1_grad[j * n_inputs + i];
        }
    }

    this->zero_gradients();
}