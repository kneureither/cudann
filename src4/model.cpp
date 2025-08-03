#include "model.h"
#include "tensor.h"



//// MODEL CLASSES ////

SingleLayerModel::SingleLayerModel(int input_size, int output_size) {
    LinearLayer linear_layer = LinearLayer(input_size, output_size);
    this->layers.push_back(linear_layer);

    this->loss = new SoftmaxCrossEntropyLoss(input_size);
}

SingleLayerModel::~SingleLayerModel() {
    delete this->loss;
}





//// LAYER CLASSES ////

LinearLayer::LinearLayer(int input_size, int output_size) {
    this->weights = Tensor<float>({input_size, output_size});
    this->biases = Tensor<float>({output_size});
    this->weights_grad = Tensor<float>({input_size, output_size});
    this->biases_grad = Tensor<float>({output_size});

    this->weights.random_init();
    this->biases.zeros();
    this->weights_grad.zeros();
    this->biases_grad.zeros();
}

Tensor<float> LinearLayer::forward(Tensor<float> input) {
    this->input = input;
    this->output = this->weights.matmul(input);
    this->output += this->biases;
    return this->output;
}

Tensor<float> LinearLayer::backward(Tensor<float> dOut) {
    this->biases_grad = dOut;
    this->weights_grad += this->input.matmul(dOut);
    return this->weights.matmul(dOut);
}

void LinearLayer::update_weights(float learning_rate) {
    this->weights_grad *= learning_rate;
    this->weights -= this->weights_grad;
    this->biases_grad *= learning_rate;
    this->biases -= this->biases_grad;
}

void LinearLayer::reset_gradients() {
    this->weights_grad.zeros();
    this->biases_grad.zeros();
}


float SoftmaxCrossEntropyLoss::forward(Tensor<float> input, Tensor<float> y_true) {
    this->y_true = y_true;
    this->logits = input
}

Tensor<float> SoftmaxCrossEntropyLoss::backward(float ) {
    // Gradient of loss w.r.t. logits
    int batch_size = this->logits.shape[0];
    Tensor<float> dZ = (this->probs - this->y_true) / batch_size;
    return dZ;
}










