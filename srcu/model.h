#ifndef MODEL_H
#define MODEL_H

#include "tensor.cuh"
#include "utils.h"


enum class Reduction {Mean, Sum};

////  LOSS CLASSES ////
template<typename T>
class Loss
{
public:
    virtual T forward(const Tensor<T>& logits, const Tensor<int>& targets) = 0;
    virtual Tensor<T> backward() = 0;
};

template<typename T>
class SoftmaxCrossEntropy : public Loss<T>  
{
    Tensor<T> props; // softmax output
public:
    explicit SoftmaxCrossEntropy(Reduction red = Reduction::Mean) 
        : reduction_(red) {};

    T forward(const Tensor<T>& logits, const Tensor<int>& targets) override {
        if (logits.shape.size() != 2) {
            throw std::invalid_argument("Loss forward: logits must be 2D [B,C]");
        }
        if (targets.shape.size() != 1) {
            throw std::invalid_argument("Loss forward: targets must be 1D [B]");
        }
        B_ = logits.shape[0];
        C_ = logits.shape[1];
        if (targets.shape[0] != B_) {
            throw std::invalid_argument("Loss forward: targets length != batch size");
        }

        // log_probs for stable NLL
        Tensor<T> log_probs = logits.log_softmax_axis1(); // [B,C]
        // also cache probs for backward
        cached_probs_ = log_probs.exp(); // [B,C]
        cached_targets_ = targets; // [B]

        // Compute NLL: loss_i = -log_probs[i, targets[i]]
        T total = T(0);
        for (size_t i = 0; i < B_; ++i)
        {
            int yi = cached_targets_[static_cast<int>(i)]; // targets is 1D int tensor
            if (yi < 0 || static_cast<size_t>(yi) >= C_)
            {
                throw std::out_of_range("targets contain class out of range");
            }
            total += -log_probs(static_cast<int>(i), yi);
        }

        if (reduction_ == Reduction::Mean)
        {
            last_loss_ = total / static_cast<T>(B_);
        }
        else
        { // Reduction::Sum
            last_loss_ = total;
        }
        return last_loss_;
    };

    Tensor<T> backward() override {
        // grad = probs
        Tensor<T> grad = cached_probs_; // copy [B,C]

        // subtract 1 at the true class for each row
        for (int i = 0; i < B_; ++i)
        {
            int yi = cached_targets_[i];
            grad(i, yi) -= T(1);
        }

        // scale for reduction
        if (reduction_ == Reduction::Mean)
        {
            grad *= (T(1) / static_cast<T>(B_));
        }
        return grad;
    };

private:
    Reduction reduction_;
    Tensor<T> cached_probs_; // [B, C] softmax(logits)
    Tensor<int> cached_targets_; // [B]
    size_t B_{0}, C_{0};
    T last_loss_{0};
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

template <typename T>
class Activation : public Layer<T>
{
public:
    Tensor<T> forward(const Tensor<T> &in) override = 0;
    Tensor<T> backward(const Tensor<T> &grad_out) override = 0;
    virtual void step(T lr) override {};
    Tensor<T> output;
};

template<typename T>
class ReLu : public Activation<T>
{
public: 
    ReLu() : input_size_(0), mask() {};
    ReLu(size_t input_size) : input_size_(input_size) {};
    ~ReLu() {};

    Tensor<T> forward(const Tensor<T> &in) override;
    Tensor<T> backward(const Tensor<T> &grad_out) override;
    virtual void step(float lr) override {return;};

    size_t get_input_size() const override { return input_size_; }
    size_t get_output_size() const override { return input_size_; }

private:
    size_t input_size_;
    Tensor<T> mask;
};

template <typename T>
Tensor<T> ReLu<T>::forward(const Tensor<T> &in)
{
    // shape of in: [batch_size, input_size]
    this->mask = (in > 0);
    this->output = in;
    this->output *= mask;
    return this->output;
}

template <typename T>
Tensor<T> ReLu<T>::backward(const Tensor<T> &grad_out)
{
    // grad_out: [batch_size, output_size]
    Tensor<T> dX = grad_out;
    dX *= mask;
    return dX;
}

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
    b_.zeros();
    W_grad_.zeros();
    b_grad_.zeros();
}

template <typename T>
Tensor<T> Linear<T>::forward(const Tensor<T> &in)
{
    // shape of in: [batch_size, input_size]
    // logger("Linear forward pass with input: \n" + in.to_string(), "DEBUG", __FILE__, __LINE__);
    this->input_cache_ = in; // Cache the input for backward pass
    // logger("Linear forward pass input_cache_: \n" + input_cache_.to_string(), "DEBUG", __FILE__, __LINE__);
    this->output = this->input_cache_.matmul(this->W_);
    //logger("Linear forward pass output: \n" + this->output.to_string(), "DEBUG", __FILE__, __LINE__);
    //logger("Linear forward pass bias \n: " + this->b_.to_string(), "DEBUG", __FILE__, __LINE__);
    this->output += this->b_;
    //logger("Linear forward pass output after adding bias: \n" + this->output.to_string(), "DEBUG", __FILE__, __LINE__);
    //logger("Linear forward pass output shape: " + this->output.shape_to_string(), "DEBUG", __FILE__, __LINE__);
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
    //logger("grad values: " + grad.to_string());
    //logger("grad out values: " + grad_out.to_string());
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
