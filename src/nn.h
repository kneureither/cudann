#ifndef NN_H
#define NN_H

class NeuralNetwork
{
public:
    double *w1;
    double *b1;
    double *w1_grad;
    double *b1_grad;
    double *output;

    int n_inputs;
    int n_outputs;

    NeuralNetwork(int n_inputs, int n_outputs);
    ~NeuralNetwork();

    void randomize_weights();
    double inline activation_function(double x);
    void neural_network_softmax(double *activations, int length);
    void forward(double *input);
    void backprop(double *input, double *target);
    void update_weights(double learning_rate);
    double loss_function(double *target);
    void zero_gradients();
    void update_weights_sgd(double learning_rate);
};

#endif // NN_H