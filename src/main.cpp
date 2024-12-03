#include "nn.h"
#include "dataset.h"

#include <iostream>

int main()
{
    // std::cout << "map_uint32(0x00000801): " << map_uint32(0x00000801) << std::endl;
    std::cout << "__BYTE_ORDER__ " << __BYTE_ORDER__ << std::endl;
    std::cout << "__ORDER_LITTLE_ENDIAN__ " << __ORDER_LITTLE_ENDIAN__ << std::endl;

    NeuralNetwork nn = NeuralNetwork(20, 5);
    nn.randomize_weights();

    std::cout << "weights: " << std::endl;
    for (int i = 0; i < nn.n_inputs; i++)
    {
        for (int j = 0; j < nn.n_outputs; j++)
        {
            std::cout << nn.w1[i * nn.n_outputs + j] << " ";
        }
        std::cout << std::endl;
    }

    Dataset dataset = Dataset();
    std::cout << "Dataset test size: " << dataset.get_test_size() << " Dataset train size: " << dataset.get_train_size() << std::endl;

    int index = 1;
    mnist_image_t *sample = dataset.get_train_sample(index);
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++)
    {
        std::cout << (int)(sample->pixels[i] > 0) << " ";
    }
    std::cout << std::endl;
    std::cout << "Sample label: " << (int)dataset.get_train_label(index) << std::endl;

    return 0;
}