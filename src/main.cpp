#include "nn.h"
#include "dataset.h"

#include <iostream>
#include <cstdint>

int main()
{
    const int max_epochs = 10;
    const int batch_size = 8;
    const float lr = 0.01;

    NeuralNetwork nn = NeuralNetwork(MNIST_IMAGE_SIZE, MNIST_LABELS);
    Dataset dataset = Dataset();
    std::cout << "Dataset test size: " << dataset.get_test_size() << " Dataset train size: " << dataset.get_train_size() << std::endl;

    int correct = 0;
    int total = 0;

    mnist_image_t *sample;
    u_int8_t *input;
    u_int8_t label;

    nn.randomize_weights();
    for (int i = 0; i < dataset.get_test_size(); i++)
    {
        sample = dataset.get_test_sample(i);
        input = sample->pixels;
        label = dataset.get_test_label(i);
        nn.forward(input);
        // std::cout << " output: ";
        // for (int j = 0; j < nn.n_outputs; j++)
        // {
        //     std::cout << nn.output[j] << " ";
        // }
        // std::cout << " predicted: " << nn.get_class_from_activations() << " " << std::endl;
        if (nn.get_class_from_activations() == label)
        {
            correct++;
        }
        total++;
    }
    std::cout << "Accuracy before training (test): " << (float)correct / total << std::endl;
    nn.randomize_weights();

    for (int epoch = 1; epoch <= max_epochs; epoch++)
    {
        int correct = 0;
        int total = 0;
        std::cout << "Epoch " << epoch << " / " << max_epochs << " : ";
        for (int batch = 0; batch < dataset.get_train_size() / batch_size; batch++)
        {
            nn.zero_gradients();
            for (int i = 0; i < batch_size; i++)
            {
                input = dataset.get_train_sample(batch * batch_size + i)->pixels;
                label = dataset.get_train_label(batch * batch_size + i);

                nn.forward(input);
                nn.backprop(input, label);
            }
            nn.update_weights_sgd(lr);

            // test accuracy
            correct = 0;
            total = 0;

            for (int i = 0; i < 1000; i++)
            {
                sample = dataset.get_test_sample(i);
                input = sample->pixels;
                label = dataset.get_test_label(i);
                nn.forward(input);
                if (nn.get_class_from_activations() == label)
                {
                    correct++;
                }
                total++;
            }
            //std::cout << " batch " << batch << " accuracy: " << (float)correct / total << " " << std::endl;
        }

        // std::cout << " done! ";

        correct = 0;
        total = 0;

        for (int i = 0; i < dataset.get_train_size(); i++)
        {
            sample = dataset.get_train_sample(i);
            input = sample->pixels;
            label = dataset.get_train_label(i);
            nn.forward(input);
            if (nn.get_class_from_activations() == label)
            {
                correct++;
            }
            total++;
        }
        std::cout << " train accuracy: " << (float)correct / total << " ";

        correct = 0;
        total = 0;

        for (int i = 0; i < dataset.get_test_size(); i++)
        {
            sample = dataset.get_test_sample(i);
            input = sample->pixels;
            label = dataset.get_test_label(i);
            nn.forward(input);
            if (nn.get_class_from_activations() == label)
            {
                correct++;
            }
            total++;
        }
        std::cout << " test accuracy: " << (float)correct / total << std::endl;
    }

    return 0;
}