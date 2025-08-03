#include "model.h"
#include "tensor.h"



//// MODEL CLASSES ////

template<typename T>
Model<T>::Model() {
    // Initialize the model with an empty layer list
    layers = std::vector<std::unique_ptr<Layer<T>>>();
}

template<typename T>
Model<T>::~Model() {
    // Clean up the layers
    for (auto& layer : layers) {
        layer.reset();
    }
    layers.clear();
}

template<typename T>
Tensor<T> Model<T>::forward(const Tensor<T>& in) {
    Tensor<T> output = in;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

template<typename T>
void Model<T>::backward(const Tensor<T>& grad) {
    Tensor<T> grad_out = grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad_out = (*it)->backward(grad_out);
    }
}

template<typename T>
void Model<T>::step(float lr) {
    for (auto& layer : layers) {
        layer->step(lr);
    }
}




//// LAYER CLASSES ////
    
template<typename T>
Linear<T>::Linear(size_t input_size, size_t output_size)
    : input_size_(input_size)
    , output_size_(output_size)
    , W_(Tensor<T>({input_size, output_size}))
    , b_(Tensor<T>({output_size}))
    , W_grad_(Tensor<T>({input_size, output_size}))
    , b_grad_(Tensor<T>({output_size}))
{
    W_.random(); // TODO maybe move to other initialization method at some point
    b_.zeros();
    W_grad_.zeros();
    b_grad_.zeros();
}

template<typename T>
Tensor<T> Linear<T>::forward(const Tensor<T>& in) {
    this->input_cache_ = in; // Cache the input for backward pass
    this->output = this->input_cache_.matmul(this->W_);
    this->output += this->b_;
    return this->output;
}

template<typename T>
Tensor<T> Linear<T>::backward(const Tensor<T>& grad_out) {
    // grad_out: [batch_size, output_size]

    // 1) gradients w.r.t. W and b
    //    dW = X^T · dZ,   where X^T: [in, batch] and dZ: [batch, out] → [in, out]
    this->W_grad_ = this->input_cache_.transpose().matmul(grad_out);

    //    db = sum(dZ, axis=0) → [out]
    this->b_grad_ = grad_out.sum(0);

    // 2) gradient w.r.t. input
    //    dX = dZ · W^T,   where dZ: [batch, out] and W^T: [out, in] → [batch, in]
    Tensor<T> dX = grad_out.matmul(this->W_.transpose());
    return dX;
}

template<typename T>
void Linear<T>::step(float lr) {
    this->W_grad_ *= lr;
    this->b_grad_ *= lr;
    // Update weights and biases
    // W -= lr * dW, b -= lr * db
    this->W_ -= this->W_grad_;
    this->b_ -= this->b_grad_;
}


// float SoftmaxCrossEntropyLoss::forward(Tensor<float> input, Tensor<float> y_true) {
//     this->y_true = y_true;
//     this->logits = input;
// }

// Tensor<float> SoftmaxCrossEntropyLoss::backward(float ) {
//     // Gradient of loss w.r.t. logits
//     int batch_size = this->logits.shape[0];
//     Tensor<float> dZ = (this->probs - this->y_true) / batch_size;
//     return dZ;
// }

template class Model<float>;
template class Linear<float>;
