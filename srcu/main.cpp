/*Training Script for the model*/

#define BATCH_SIZE 32
#define MAX_EPOCHS 1
#define LOG_LEVEL "INFO"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>

#include "model.h"
#include "dataloader.h"

#ifdef CUDA_AVAILABLE
#include "tensor.cuh"
#else
#include "tensor.h"
#endif

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
    int log_every_steps = 100;
    size_t eval_samples = 1000;

    // setup timing
    std::clock_t start_time = std::clock();
    double dataloader_time=0.0;
    double allocation_time=0.0;
    double duration = 0.0;
    double data_loading_time = 0.0;
    double forward_time = 0.0;
    double backward_time = 0.0;
    double step_time = 0.0;
    double eval_time = 0.0;
    double epoch_time = 0.0;


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
    Tensor<int> train_labels = train_loader.load_labels();
    Tensor<precision> test_data = test_loader.load_data();
    Tensor<int> test_labels = test_loader.load_labels();
    logger(train_data.shape_to_string(), "INFO", __FILE__, __LINE__);
    logger(test_data.shape_to_string(), "INFO", __FILE__, __LINE__);
    
    // allocate batch memory
    Tensor<precision> data_batch({BATCH_SIZE, MNIST_IMAGE_SIZE});
    Tensor<int> label_batch({BATCH_SIZE});
    Tensor<precision> logits({BATCH_SIZE, MNIST_LABELS});
    Tensor<precision> d_logits({BATCH_SIZE, MNIST_LABELS});

    // setup model
    Model<precision> model;
    model.layers.push_back(std::make_unique<Linear<precision>>(MNIST_IMAGE_SIZE, MNIST_LABELS));
    //model.layers.push_back(std::make_unique<ReLu<precision>>(30));
    //model.layers.push_back(std::make_unique<Linear<precision>>(30, MNIST_LABELS));

    // setup loss
    SoftmaxCrossEntropy<precision> loss_fn(Reduction::Mean);

    #ifdef CUDA_AVAILABLE
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);   // pick your device id
    printf("warpSize=%d\n", p.warpSize);                      // usually 32
    printf("maxThreadsPerBlock=%d\n", p.maxThreadsPerBlock);  // often 1024
    printf("maxThreadsPerMultiProcessor=%d\n", p.maxThreadsPerMultiProcessor);
    printf("sharedMemPerMultiprocessor=%zu\n", p.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor=%d\n", p.regsPerMultiprocessor);
    printf("multiProcessorCount=%d\n", p.multiProcessorCount);
    #endif


    for (int epoch = 1; epoch <= max_epochs; ++epoch)
    {
        logger("Epoch " + std::to_string(epoch), "INFO", __FILE__, __LINE__);

        for (int batch_idx = 1; batch_idx < train_loader.get_max_num_batches(); ++batch_idx)
        {
            // Load a batch of data
            std::clock_t batch_start_time = std::clock();
            data_batch.view(train_data, BATCH_SIZE * (batch_idx-1), BATCH_SIZE);
            logger("data_batch : " + data_batch.to_string(), "DEBUG", __FILE__, __LINE__);
            logger("data_batch shape: " + data_batch.shape_to_string(), "DEBUG", __FILE__, __LINE__);

            label_batch.view(train_labels, BATCH_SIZE * (batch_idx-1), BATCH_SIZE);
            logger("Batch labels: " + label_batch.to_string(), "DEBUG", __FILE__, __LINE__);
            logger("Label_batch shape: " + label_batch.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            data_loading_time += (std::clock() - batch_start_time) / (double) CLOCKS_PER_SEC;

            // Forward pass
            std::clock_t forward_start_time = std::clock();
            logits = model.forward(data_batch);
            logger("Logits: " + logits.to_string(), "DEBUG", __FILE__, __LINE__);

            precision loss = loss_fn.forward(logits, label_batch);
            logger(" -- Batch idx: " + std::to_string(batch_idx) + " Loss: " + std::to_string(loss), "DEBUG", __FILE__, __LINE__);
            forward_time += (std::clock() - forward_start_time) / (double) CLOCKS_PER_SEC;

            // Backward pass
            std::clock_t backward_start_time = std::clock();
            d_logits = loss_fn.backward(); // ∂ℓ/∂logits
            logger("d_logits shape: " + d_logits.shape_to_string(), "DEBUG", __FILE__, __LINE__);
            logger("d_logits values: " + d_logits.to_string(), "DEBUG", __FILE__, __LINE__);
            model.backward(d_logits); // backprop through all layers
            backward_time += (std::clock() - backward_start_time) / (double) CLOCKS_PER_SEC;

            // Update model parameters
            std::clock_t step_start_time = std::clock();
            model.step(learning_rate);
            step_time += (std::clock() - step_start_time) / (double) CLOCKS_PER_SEC;

            if (batch_idx % log_every_steps == 0) {
                logger("Batch " + std::to_string(batch_idx) + " Loss: " + std::to_string(loss), "INFO");
            }

            // eval
            std::clock_t eval_start_time = std::clock();
            if (batch_idx % eval_every_steps == 0) {
                logger("Evaluating model on test set...", "INFO");
                eval_result res = evaluation(model, test_loader, false);
                logger("Evaluation results: " + std::to_string(res.num_correct) + "/" + std::to_string(res.num_total) + " (Accuracy: " + std::to_string(res.accuracy) + ")", "INFO");
                }
            eval_time += (std::clock() - eval_start_time) / (double) CLOCKS_PER_SEC;

            //if(batch_idx == 1) break;
        }
    }

    duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    logger("Final Evaluation...", "INFO");
    eval_result res = evaluation(model, test_loader, false);
    logger("Evaluation results: " + std::to_string(res.num_correct) + "/" + std::to_string(res.num_total) + " (Accuracy: " + std::to_string(res.accuracy) + ")", "INFO");


    logger("==================================", "INFO");
    logger("Batch size: " + std::to_string(BATCH_SIZE), "INFO");
    logger("Max epochs: " + std::to_string(max_epochs), "INFO");
    logger("Learning rate: " + std::to_string(learning_rate), "INFO");
    # if CUDA_AVAILABLE
    logger("Using CUDA device: " + std::string(p.name), "INFO");
    logger("Compute capability: " + std::to_string(p.major) + "." + std::to_string(p.minor), "INFO");
    logger("Total global memory: " + std::to_string(p.totalGlobalMem / (1024 * 1024)) + " MB", "INFO");
    logger("Shared memory per block: " + std::to_string(p.sharedMemPerBlock / 1024) + " KB", "INFO");
    logger("Registers per block: " + std::to_string(p.regsPerBlock), "INFO");
    logger("Warp size: " + std::to_string(p.warpSize), "INFO");
    logger("Max threads per block: " + std::to_string(p.maxThreadsPerBlock), "INFO");
    logger("Max threads per multiprocessor: " + std::to_string(p.maxThreadsPerMultiProcessor), "INFO");
    logger("Number of multiprocessors: " + std::to_string(p.multiProcessorCount), "INFO");
    # else
    logger("Using CPU", "INFO");
    # endif

    logger("Training completed in " + std::to_string(duration) + " seconds", "INFO");
    logger("Data loading time: " + std::to_string(data_loading_time) + " seconds (" + std::to_string(data_loading_time / duration * 100) + "%)", "INFO");
    logger("Forward pass time: " + std::to_string(forward_time) + " seconds (" + std::to_string(forward_time / duration * 100) + "%)", "INFO");
    logger("Backward pass time: " + std::to_string(backward_time) + " seconds (" + std::to_string(backward_time / duration * 100) + "%)", "INFO");
    logger("Step time: " + std::to_string(step_time) + " seconds (" + std::to_string(step_time / duration * 100) + "%)", "INFO");
    logger("Evaluation time: " + std::to_string(eval_time) + " seconds (" + std::to_string(eval_time / duration * 100) + "%)", "INFO");

    return 0;
}