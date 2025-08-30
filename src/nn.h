#ifndef NN_H
#define NN_H

#include <cstdint>

class NeuralNetwork
{
public:
    double *w1;
    double *b1;
    double *w1_grad;
    double *b1_grad;
    double *output;
    uint8_t *logits;

    int n_inputs;
    int n_outputs;

    NeuralNetwork(int n_inputs, int n_outputs);
    ~NeuralNetwork();

    void randomize_weights();
    void neural_network_softmax(double *activations, int length);
    void forward(uint8_t *input);
    void backprop(uint8_t *input, u_int8_t label);
    void update_weights(double learning_rate);
    void zero_gradients();
    void update_weights_sgd(double learning_rate);
    int get_class_from_activations();
    uint8_t *get_logits_from_label(uint8_t label);
};

#endif // NN_H