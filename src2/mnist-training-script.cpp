#include "neural_network.h"
#include "mnist_loader.h"
#include <iostream>
#include <random>
#include <chrono>

int main() {
    // MNIST Data Loading
    MNISTLoader mnist_loader;
    mnist_loader.load_train_data("dat/train-images-idx3-ubyte", "dat/train-labels-idx1-ubyte");
    mnist_loader.load_test_data("dat/t10k-images-idx3-ubyte", "dat/t10k-labels-idx1-ubyte");

    Tensor train_images = mnist_loader.get_training_images();
    Tensor train_labels = mnist_loader.get_training_labels();
    Tensor test_images = mnist_loader.get_test_images();
    Tensor test_labels = mnist_loader.get_test_labels();

    std::cout << "Train images shape: " << train_images.data().size() << std::endl;
    std::cout << "Train labels shape: " << train_labels.data().size() << std::endl;
    std::cout << "Test images shape: " << test_images.data().size() << std::endl;
    std::cout << "Test labels shape: " << test_labels.data().size() << std::endl;


    // Network Configuration
    NeuralNetwork model;

    // Add layers
    model.add_layer(std::make_unique<DenseLayer>(
        784,    // Input size (28x28 flattened)
        10,    // Hidden layer size
        std::make_unique<Softmax>()
    ));


    // model.add_layer(std::make_unique<DenseLayer>(
    //     128,    // Previous layer output size
    //     64,     // Next hidden layer size
    //     std::make_unique<ReLU>()
    // ));

    // model.add_layer(std::make_unique<DenseLayer>(
    //     64,     // Previous layer output size
    //     10,     // Output size (10 digit classes)
    //     std::make_unique<Sigmoid>()
    // ));

    // Configure loss function
    model.set_loss_function(std::make_unique<CrossEntropyLoss>());

    // Training Hyperparameters
    const float learning_rate = 0.01f;
    const int epochs = 5;
    const size_t batch_size = 8;

    // // Random batch sampling
    // std::random_device rd;
    // std::mt19937 gen(rd());

    // Training Loop
    auto training_start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        int correct_predictions = 0;

        // Mini-batch training
        //for (auto batch_start : tqdm::range(size_t(0), train_images.shape()[0], batch_size))
        for (size_t batch_start = 0; batch_start < train_images.shape()[0]; batch_start += batch_size)
        //  for (size_t batch_start = 0; batch_start < 50; batch_start += batch_size)
        {
            if (batch_start % 100 == 0) {
                std::cout << "\r[Training] Batch: " << batch_start << " / " << train_images.shape()[0] << " " << std::flush;
            }

            for (int batch_idx = 0;
                 batch_idx < batch_size && (batch_start + batch_idx) < train_images.shape()[0];
                 ++batch_idx)
            {

                size_t sample_idx = batch_start + batch_idx;

                // Extract single image and label
                Tensor input({1, 784});
                Tensor target({1, 10});

                std::copy_n(train_images.data().begin() + sample_idx * 784, 784, input.data().begin());
                std::copy_n(train_labels.data().begin() + sample_idx * 10, 10, target.data().begin());

                // Forward pass
                Tensor prediction = model.predict(input);

                // Compute loss
                float loss = model.get_loss_function().compute(prediction, target);
                total_loss += loss;

                // Backward pass (update weights)
                model.train(input, target, learning_rate);

                // Simple accuracy tracking
                int predicted_class = std::distance(
                    prediction.data().begin(),
                    std::max_element(prediction.data().begin(), prediction.data().end()));
                int true_class = std::distance(
                    target.data().begin(),
                    std::max_element(target.data().begin(), target.data().end()));

                if (predicted_class == true_class)
                {
                    correct_predictions++;
                }
            }
            }
        // Epoch summary
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time);
        
        float accuracy = static_cast<float>(correct_predictions) / train_images.shape()[0];
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 
                  << ": Loss = " << total_loss 
                  << ", Accuracy = " << accuracy * 100.0f << "%"
                  << ", Time = " << epoch_duration.count() << "s" << std::endl;

        // Optional: Add early stopping or learning rate decay
    }

    // Calculate total training time
    auto training_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(training_end_time - training_start_time);
    std::cout << "Total training time: " << total_duration.count() << " minutes" << std::endl;

    // Model Evaluation
    int test_correct_predictions = 0;
    for (size_t i = 0; i < test_images.shape()[0]; ++i) {
        Tensor input({1, 784});
        Tensor target({1, 10});
        
        std::copy_n(test_images.data().begin() + i * 784, 784, input.data().begin());
        std::copy_n(test_labels.data().begin() + i * 10, 10, target.data().begin());

        Tensor prediction = model.predict(input);

        int predicted_class = std::distance(
            prediction.data().begin(), 
            std::max_element(prediction.data().begin(), prediction.data().end())
        );
        int true_class = std::distance(
            target.data().begin(), 
            std::max_element(target.data().begin(), target.data().end())
        );

        if (predicted_class == true_class) {
            test_correct_predictions++;
        }
    }

    float test_accuracy = static_cast<float>(test_correct_predictions) / test_images.shape()[0];
    std::cout << "Test Accuracy: " << test_accuracy * 100.0f << "%" << std::endl;

    return 0;
}
