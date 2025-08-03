#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

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
    Layer();
    virtual ~Layer();

    virtual Tensor<T> forward(const Tensor<T>& in) = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_out) = 0;
    virtual void step(float lr) = 0;

    // only to store the input and output tensors -> optional
    Tensor<T> input;
    Tensor<T> output;
};

template<typename T>
class Linear : public Layer<T>
{
public:
    Linear(size_t input_size, size_t output_size);
    ~Linear();

    virtual Tensor<T> forward(const Tensor<T> &in) override;
    virtual Tensor<T> backward(const Tensor<T> &grad_out) override;
    virtual void step(float lr) override;
    void reset_gradients();

private:
    size_t input_size_;
    size_t output_size_;
    Tensor<T> W_, b_;
    Tensor<T> input_cache_;
    Tensor<T> W_grad_, b_grad_;
};

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

#endif
