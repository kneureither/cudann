/*Training Script for the model*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>

#include "model.h"
#include "dataloader.h"
#include "tensor.h"
#include "utils.h"

bool test_tensor_creation() {
    logger("test_tensor_creation");
    Tensor<float> a({2, 3});
    a.zeros();
    std::cout << "2-dim tensor: \n" << a.to_string() << std::endl;
    return true;
}

bool test_tensor_dot_1d() {
    logger("test_tensor_dot_1d");
    Tensor<float> a({3, 2});
    for (int i = 0; i < a.get_size(); i++) {
        a.get_data_ptr()[i] = i;
    }
    std::cout << "2-dim tensor: \n" << a.to_string() << std::endl;
    
    Tensor<float> b({2});
    for (int i = 0; i < b.get_size(); i++) {
        b.get_data_ptr()[i] = i;
    }
    std::cout << "1-dim tensor: \n" << b.to_string() << std::endl;

    Tensor<float> c = a.dot_2d(b);
    logger("a.shape: " + a.shape_to_string() + " b.shape: " + b.shape_to_string() + " c.shape: " + c.shape_to_string());
    std::cout << "1-dim tensor: \n" << c.to_string() << std::endl;
    return true;
}

bool test_tensor_matmul() {
    logger("test_tensor_matmul");
    Tensor<float> a({2, 3});
    for (int i = 0; i < a.get_size(); i++) {
        a.get_data_ptr()[i] = i+1;
    }
    std::cout << "LEFT MATRIX: \n" << a.to_string() << std::endl;

    Tensor<float> b({3, 2});
    b.zeros();
    b[0, 0] =   b[0, 1] = 1;
    b[1, 0] =   b[1, 1] = 2;
    b[2, 0] =   b[2, 1] = 3;

    std::cout << "RIGHT MATRIX: \n" << b.to_string() << std::endl;

    Tensor<float> c = a.matmul(b);
    std::cout << "RESULT: \n" << c.to_string() << std::endl;

    if (c.shape[0] != 2 || c.shape[1] != 2) {
        return false;
    }

    if (c[0, 0] != 1*1 + 2*2 + 3*3) return false;
    else if (c[1, 0] != 1 * 4 + 2 * 5 + 3 * 6) return false;
    else if (c[0, 1] != c[0, 0]) return false;
    else if (c[1, 1] != c[1, 0]) return false;
    
    return true;
}

bool test_dataloader() {
    DataLoader train_loader = DataLoader("../dat/train-images-idx3-ubyte", "../dat/train-labels-idx1-ubyte", 2);
    std::cout << "Train loader size: " << train_loader.get_size() << std::endl;
    std::cout << "Train loader lable (0): " << (int) train_loader.get_label(0) << std::endl;
    std::cout << "Train loader image (0): " << std::endl << train_loader.get_image_as_string(0) << std::endl;

    Tensor<float> data = train_loader.load_data_batch(0);
    Tensor<int> labels = train_loader.load_labels_batch(0);
    std::cout << "Data: " << data.shape_to_string() << std::endl;
    std::cout << "Labels: " << labels.shape_to_string() << std::endl;

    std::cout << "Data: " << data.to_string() << std::endl;
    std::cout << "Labels: " << labels.to_string() << std::endl;
    return true;
}

int main()
{
    test_tensor_creation();
    assert(test_tensor_matmul());
    assert(test_dataloader());
}
