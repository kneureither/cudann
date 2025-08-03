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

    virtual Tensor<T> forward(Tensor<T> input) = 0;
    virtual Tensor<T> backward(Tensor<T> dOut) = 0;

    Tensor<T> input;
    Tensor<T> output;
};

template<typename T>
class LinearLayer : public Layer<T>
{
public:
    LinearLayer(int input_size, int output_size);
    ~LinearLayer();

    Tensor<float> forward(Tensor<float> input);
    Tensor<float> backward(Tensor<float> dOut);
    void update_weights(float learning_rate);
    void reset_gradients();

private:
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> weights_grad;
    Tensor<float> biases_grad;
};

//// MODEL CLASSES ////

template<typename T>
class Model
{
public:
    Model();
    ~Model();

    // for training
    virtual Tensor<T> forward(Tensor<T> input);
    virtual Tensor<T> backward(Tensor<T> output, Tensor<T> dOut);
    virtual void update_weights(float learning_rate);

    // for validation
    virtual Tensor<int> predict(Tensor<T> input);
    std::vector<Layer> layers;
    Loss *loss;
};

template<typename T>
class SingleLayerModel : public Model<T>
{
public:
    SingleLayerModel(int input_size, int output_size);
    ~SingleLayerModel();
};

#endif
