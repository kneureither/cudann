/*Training Script for the model*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "model.h"
#include "dataloader.h"
#include "tensor.h"

#define BATCH_SIZE 8

int main()
{

    // load data
    DataLoader train_loader = DataLoader(
        "../dat/train-images-idx3-ubyte",
        "../dat/train-labels-idx1-ubyte",
        BATCH_SIZE);

    DataLoader test_loader = DataLoader(
        "../dat/t10k-images-idx3-ubyte",
        "../dat/t10k-labels-idx1-ubyte",
        BATCH_SIZE);

    Tensor<float> train_data = train_loader.load_data();
    Tensor<float> test_data = test_loader.load_data();

    logger(train_data.shape_to_string(), "INFO", __FILE__, __LINE__);
    logger(test_data.shape_to_string(), "INFO", __FILE__, __LINE__);

    // initialize model
    // setup timing
    // train loop

    return 0;
}