/*Training Script for the model*/

#define BATCH_SIZE 8
#define MAX_EPOCHS 1
#define LOG_LEVEL "DEBUG"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>

#include "model.h"
#include "dataloader.h"
#include "tensor.cuh"

using precision = float;

struct eval_result {
    unsigned int num_total=0;
    unsigned int num_correct=0;
    float accuracy = 0.0f;
};

eval_result evaluation(Model<precision> &model, DataLoader &test_loader, bool full_set_eval) {
    unsigned int num_correct = 0;
    unsigned int num_total = 0;

    Tensor<precision> test_batch;
    Tensor<int> test_labels;

    if (full_set_eval) {
        test_batch = test_loader.load_data();
        test_labels = test_loader.load_labels();
    }
    else {
        test_batch = test_loader.load_data_batch(0);
        test_labels = test_loader.load_labels_batch(0);
    }

    // get predictions
    Tensor<precision> logits = model.forward(test_batch);
    Tensor<int> predicted_batch = logits.argmax(/*axis=*/ 1);

    for (int i = 0; i < test_batch.get_shape()[0]; ++i)
    {
        int actual = test_labels[i];
        int predicted = predicted_batch[i];
        if (predicted == actual)
        {
            num_correct++;
        }
        num_total++;
    }

    float accuracy = static_cast<float>(num_correct) / num_total;

    return eval_result({num_total, num_correct, accuracy});
};

int main() {

    int max_epochs = MAX_EPOCHS;
    float learning_rate = 0.001f;
    int eval_every_steps = 1000;
    int log_evey_steps = 100;
    size_t eval_samples = 1000;


    // load data
    DataLoader train_loader = DataLoader(
        "dat/train-images-idx3-ubyte",
        "dat/train-labels-idx1-ubyte",
        BATCH_SIZE);

    DataLoader test_loader = DataLoader(
        "dat/t10k-images-idx3-ubyte",
        "dat/t10k-labels-idx1-ubyte",
        eval_samples);

    Tensor<precision> train_data = train_loader.load_data();
    Tensor<precision> test_data = test_loader.load_data();
    logger(train_data.shape_to_string(), "INFO", __FILE__, __LINE__);
    logger(test_data.shape_to_string(), "INFO", __FILE__, __LINE__);

    // setup model
    Model<precision> model;
    model.layers.push_back(std::make_unique<Linear<precision>>(MNIST_IMAGE_SIZE, 30));
    model.layers.push_back(std::make_unique<ReLu<precision>>(30));
    model.layers.push_back(std::make_unique<Linear<precision>>(30, MNIST_LABELS));

    // setup loss
    SoftmaxCrossEntropy<precision> loss_fn(Reduction::Mean);

    // setup timing
    

    for (int epoch = 1; epoch <= max_epochs; ++epoch)
    {
        logger("Epoch " + std::to_string(epoch), "INFO", __FILE__, __LINE__);
        for (int batch_idx = 1; batch_idx < train_loader.get_max_num_batches(); ++batch_idx)
        {
            
            // Load a batch of data
            Tensor<precision> data_batch = train_loader.load_data_batch(batch_idx-1);
            Tensor<int> label_batch = train_loader.load_labels_batch(batch_idx-1);
            //logger("Batch labels: " + label_batch.to_string(), "DEBUG", __FILE__, __LINE__);
            logger("Label_batch shape: " + label_batch.shape_to_string(), "DEBUG", __FILE__, __LINE__);

            // Forward pass
            Tensor<precision> logits = model.forward(data_batch);
            logger("Logits: " + logits.to_string(), "DEBUG", __FILE__, __LINE__);
            precision loss = loss_fn.forward(logits, label_batch);
            logger(" -- Batch idx: " + std::to_string(batch_idx) + " Loss: " + std::to_string(loss), "DEBUG", __FILE__, __LINE__);

            // Backward pass
            Tensor<precision> d_logits = loss_fn.backward(); // ∂ℓ/∂logits
            logger("d_logits shape: " + d_logits.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            logger("d_logtis values: " + d_logits.to_string(), "DEBUG", __FILE__, __LINE__);
            model.backward(d_logits); // backprop through all layers

            // Update model parameters
            model.step(learning_rate);

            if (batch_idx % log_evey_steps == 0) {
                logger("Batch " + std::to_string(batch_idx) + " Loss: " + std::to_string(loss), "INFO");
            }

            // eval
            if (batch_idx % eval_every_steps == 0) {
                logger("Evaluating model on test set...", "INFO");
                eval_result res = evaluation(model, test_loader, false);
                logger("Evaluation results: " + std::to_string(res.num_correct) + "/" + std::to_string(res.num_total) + " (Accuracy: " + std::to_string(res.accuracy) + ")", "INFO");
                }

            break;
        }
    }

    return 0;

    logger("Final Evaluation...", "INFO");
    eval_result res = evaluation(model, test_loader, false);
    logger("Evaluation results: " + std::to_string(res.num_correct) + "/" + std::to_string(res.num_total) + " (Accuracy: " + std::to_string(res.accuracy) + ")", "INFO");

    return 0;
}