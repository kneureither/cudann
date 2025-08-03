#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "utils.h"

////  LOSS CLASSES ////
class Loss
{
public:
    virtual Tensor<float> forward(Tensor<float> input, Tensor<float> y_true) = 0;
    virtual Tensor<float> backward(Tensor<float> dOut) = 0;
};

class SoftmaxCrossEntropyLoss : public Loss
{
public:
    SoftmaxCrossEntropyLoss(int input_size);
    ~SoftmaxCrossEntropyLoss();

    Tensor<float> forward(Tensor<float> input, Tensor<float> y_true);
    Tensor<float> backward(float loss);

private:
    Tensor<float> softmax;
    Tensor<float> log_softmax;
    Tensor<float> y_true;
    Tensor<float> logits;
};

//// LAYER CLASSES ////

template<typename T>
class Layer
{
public:
    Layer() {};
    virtual ~Layer() {};

    virtual Tensor<T> forward(const Tensor<T>& in) = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_out) = 0;
    virtual void step(float lr) = 0;


    // Getters for weights and biases
    virtual Tensor<T> get_weights() const { return Tensor<T>(); }
    virtual Tensor<T> get_biases() const { return Tensor<T>(); }

    // Getters for input and output sizes
    virtual size_t get_input_size() const = 0;
    virtual size_t get_output_size() const = 0;

    Tensor<T> output; // Output of the layer, used for backward pass
};

template<typename T>
class Linear : public Layer<T>
{
public:
    Linear();
    Linear(size_t input_size, size_t output_size);
    ~Linear();

    // forward: Z = X · W + b
    virtual Tensor<T> forward(const Tensor<T> &in) override;

    // backward: given dL/dZ, compute dL/dX, and store dL/dW, dL/db
    virtual Tensor<T> backward(const Tensor<T> &grad_out) override;

    // SGD step: W -= lr * dW,  b -= lr * db
    virtual void step(float lr) override;

    // Getters for weights and biases
    Tensor<T> get_weights() const override { return W_; }
    Tensor<T> get_biases() const override { return b_; }
    // Getters for sizes
    size_t get_input_size() const override { return input_size_; }
    size_t get_output_size() const override { return output_size_; }

private:
    size_t input_size_;
    size_t output_size_;
    Tensor<T> W_, b_;
    Tensor<T> input_cache_;
    Tensor<T> W_grad_, b_grad_;
};

template<typename T>
Linear<T>::Linear() : input_size_(0), output_size_(0), W_(), b_(), W_grad_(), b_grad_()
{
    // Default constructor
}

template<typename T>
Linear<T>::~Linear()
{
    // Destructor
}

template <typename T>
Linear<T>::Linear(size_t input_size, size_t output_size)
    : input_size_(input_size)
    , output_size_(output_size)
    , W_(Tensor<T>({input_size, output_size}))
    , b_(Tensor<T>({output_size}))
    , W_grad_(Tensor<T>({input_size, output_size}))
    , b_grad_(Tensor<T>({output_size}))
{
    // Xavier uniform initialization
    float limit = std::sqrt(6.0f / (input_size + output_size));
    W_.random_uniform(-limit, limit);
    logger("Linear layer initialized with weights: \n" + W_.to_string(), "DEBUG", __FILE__, __LINE__);
    b_[0] = 0.7; // Initialize bias to zero
    b_[1] = 0.5;
    W_grad_.zeros();
    b_grad_.zeros();
}

template <typename T>
Tensor<T> Linear<T>::forward(const Tensor<T> &in)
{
    // shape of in: [batch_size, input_size]
    logger("Linear forward pass with input: \n" + in.to_string(), "DEBUG", __FILE__, __LINE__);
    this->input_cache_ = in; // Cache the input for backward pass
    logger("Linear forward pass input_cache_: \n" + input_cache_.to_string(), "DEBUG", __FILE__, __LINE__);
    this->output = this->input_cache_.matmul(this->W_);
    logger("Linear forward pass output: \n" + this->output.to_string(), "DEBUG", __FILE__, __LINE__);
    logger("Linear forward pass bias \n: " + this->b_.to_string(), "DEBUG", __FILE__, __LINE__);
    this->output += this->b_;
    logger("Linear forward pass output after adding bias: \n" + this->output.to_string(), "DEBUG", __FILE__, __LINE__);
    logger("Linear forward pass output shape: " + this->output.shape_to_string(), "DEBUG", __FILE__, __LINE__);
    return this->output;
}

template <typename T>
Tensor<T> Linear<T>::backward(const Tensor<T> &grad_out)
{
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

template <typename T>
void Linear<T>::step(float lr)
{
    this->W_grad_ *= lr;
    this->b_grad_ *= lr;
    // Update weights and biases
    // W -= lr * dW, b -= lr * db
    this->W_ -= this->W_grad_;
    this->b_ -= this->b_grad_;
}

//// MODEL CLASSES ////

template<typename T>
class Model
{
public:
    Model();
    ~Model();

    // for training
    Tensor<T> forward(const Tensor<T>& in);
    void backward(const Tensor<T>& grad);
    virtual void step(float lr);

    std::vector<std::unique_ptr<Layer<T>>> layers;
};

template <typename T>
Model<T>::Model()
{
    // Initialize the model with an empty layer list
    layers = std::vector<std::unique_ptr<Layer<T>>>();
}

template <typename T>
Model<T>::~Model()
{
    // Clean up the layers
    for (auto &layer : layers)
    {
        layer.reset();
    }
    layers.clear();
}

template <typename T>
Tensor<T> Model<T>::forward(const Tensor<T> &in)
{
    Tensor<T> output = in;
    logger("Model forward pass with input shape: " + output.shape_to_string(), "DEBUG", __FILE__, __LINE__);
    for (auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

template <typename T>
void Model<T>::backward(const Tensor<T> &grad)
{
    Tensor<T> grad_out = grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        grad_out = (*it)->backward(grad_out);
    }
}

template <typename T>
void Model<T>::step(float lr)
{
    for (auto &layer : layers)
    {
        layer->step(lr);
    }
}

#endif
