#include "nn.h"
#include "dataset.h"

#include <iostream>

int main()
{
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

    return 0;
}
