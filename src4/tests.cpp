/*Training Script for the model*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <memory>

#include "model.h"
#include "dataloader.h"
#include "tensor.h"
#include "utils.h"

bool test_tensor_creation()
{
    logger("test_tensor_creation");
    Tensor<float> a({2, 3});
    a.zeros();
    std::cout << "2-dim tensor: \n"
              << a.to_string() << std::endl;
    return true;
}

// bool test_tensor_dot_1d()
// {
//     logger("test_tensor_dot_1d");
//     Tensor<float> a({3, 2});
//     for (int i = 0; i < a.get_size(); i++)
//     {
//         a.get_data_ptr()[i] = i;
//     }
//     std::cout << "2-dim tensor: \n"
//               << a.to_string() << std::endl;

//     Tensor<float> b({2});
//     for (int i = 0; i < b.get_size(); i++)
//     {
//         b.get_data_ptr()[i] = i;
//     }
//     std::cout << "1-dim tensor: \n"
//               << b.to_string() << std::endl;

//     Tensor<float> c = a.dot_2d(b);
//     logger("a.shape: " + a.shape_to_string() + " b.shape: " + b.shape_to_string() + " c.shape: " + c.shape_to_string());
//     std::cout << "1-dim tensor: \n"
//               << c.to_string() << std::endl;
//     return true;
// }

bool test_tensor_matmul()
{
    logger("test_tensor_matmul");
    Tensor<float> a({2, 3});
    for (int i = 0; i < a.get_size(); i++)
    {
        a.get_data_ptr()[i] = i + 1;
    }
    std::cout << "LEFT MATRIX: \n"
              << a.to_string() << std::endl;

    Tensor<float> b({3, 2});
    b.zeros();
    b(0, 0) = b(0, 1) = 1;
    b(1, 0) = b(1, 1) = 2;
    b(2, 0) = b(2, 1) = 3;

    std::cout << "RIGHT MATRIX: \n"
              << b.to_string() << std::endl;

    Tensor<float> c = a.matmul(b);
    std::cout << "RESULT: \n"
              << c.to_string() << std::endl;

    if (c.shape[0] != 2 || c.shape[1] != 2)
    {
        return false;
    }

    if (c(0, 0) != 1 * 1 + 2 * 2 + 3 * 3)
        return false;
    else if (c(1, 0) != 1 * 4 + 2 * 5 + 3 * 6)
        return false;
    else if (c(0, 1) != c(0, 0))
        return false;
    else if (c(1, 1) != c(1, 0))
        return false;

    return true;
}

bool test_dataloader()
{
    DataLoader train_loader = DataLoader("dat/train-images-idx3-ubyte", "dat/train-labels-idx1-ubyte", 2);
    std::cout << "Train loader size: " << train_loader.get_size() << std::endl;
    std::cout << "Train loader lable (0): " << (int)train_loader.get_label(0) << std::endl;
    std::cout << "Train loader image (0): " << std::endl
              << train_loader.get_image_as_string(0) << std::endl;

    Tensor<float> data = train_loader.load_data_batch(0);
    Tensor<int> labels = train_loader.load_labels_batch(0);
    std::cout << "Data: " << data.shape_to_string() << std::endl;
    std::cout << "Labels: " << labels.shape_to_string() << std::endl;

    std::cout << "Data: " << data.to_string() << std::endl;
    std::cout << "Labels: " << labels.to_string() << std::endl;
    return true;
}

bool test_tensor_sum()
{
    logger("test_tensor_sum");
    Tensor<float> a({2, 3});
    for (int i = 0; i < a.get_size(); i++)
    {
        a.get_data_ptr()[i] = i + 1;
    }
    std::cout << "Tensor: \n"
              << a.to_string() << std::endl;

    Tensor<float> sum2 = a.sum(0);
    std::cout << "Sum over axis 0: \n"
              << sum2.to_string() << std::endl;

    if (sum2[0] != 1 + 4 || sum2[1] != 2 + 5 || sum2[2] != 3 + 6 )
    {
        return false;
    }

    Tensor<float> sum = a.sum(1);
    std::cout << "Sum over axis 1: \n"
              << sum.to_string() << std::endl;

    if (sum[0] != 1 + 2 + 3 || sum[1] != 4 + 5 + 6)
    {
        return false;
    }

    return true;
}


bool test_model_init()
{
    logger("test_model_init");
    Model<float> model;
    model.layers.push_back(std::make_unique<Linear<float>>(3, 2));

    std::cout << "Model initialized with 1 layer: " << model.layers.size() 
              << " And input size: " << model.layers[0]->get_input_size() 
              << " And output size: " << model.layers[0]->get_output_size() << std::endl;
    Tensor<float> input({2, 3});
    for (int i = 0; i < input.get_size(); i++)
    {
        input.get_data_ptr()[i] = i + 1;
    }
    std::cout << "Input: \n"
              << input.to_string() << std::endl;

    Tensor<float> output = model.forward(input);
    std::cout << "Output: \n"
              << output.to_string() << std::endl;

    if (output.shape[0] != 2 || output.shape[1] != 2)
        return false;

    return true;
}

bool test_model_backward()
{
    logger("\n\ntest_model_backward");
    Model<float> model;
    model.layers.push_back(std::make_unique<Linear<float>>(3, 2));

    Tensor<float> input({4, 3});
    for (int i = 0; i < input.get_size(); i++)
    {
        input.get_data_ptr()[i] = i + 1;
    }
    std::cout << "Input: \n"
              << input.to_string() << std::endl;

    Tensor<float> output = model.forward(input);
    std::cout << "Output: \n"
              << output.to_string() << std::endl;

    Tensor<float> grad_out({4, 2});
    for (int i = 0; i < grad_out.get_size(); i++)
    {
        grad_out.get_data_ptr()[i] = 1.0f; // Simple gradient
    }

    std::cout << "Weights before backward pass: \n"
              << model.layers[0]->get_weights().to_string() << std::endl;

    

    
    model.backward(grad_out);
    model.step(0.01f); // Perform a step with learning rate

    std::cout << "Weights after backward pass: \n"
              << model.layers[0]->get_weights().to_string() << std::endl;
    return true;
}

bool test_tensor_log_softmax_axis1() {
    logger("test_tensor_log_softmax_axis1");
    
    // Test case 1: Simple 2x3 tensor
    Tensor<float> input({2, 3});
    input.get_data_ptr()[0] = 1.0f; input.get_data_ptr()[1] = 2.0f; input.get_data_ptr()[2] = 3.0f;
    input.get_data_ptr()[3] = 4.0f; input.get_data_ptr()[4] = 5.0f; input.get_data_ptr()[5] = 6.0f;
    
    std::cout << "Input tensor:\n" << input.to_string() << std::endl;
    
    Tensor<float> result = input.log_softmax_axis1();
    std::cout << "Log-softmax result:\n" << result.to_string() << std::endl;
    
    // Verify properties of log-softmax:
    // 1. Sum of exp(log_softmax) should be 1 for each row
    for (int i = 0; i < 2; i++) {
        float sum_exp = 0.0f;
        for (int j = 0; j < 3; j++) {
            sum_exp += std::exp(result.get_data_ptr()[i * 3 + j]);
        }
        std::cout << "Row " << i << " sum of exp(log_softmax): " << sum_exp << std::endl;
        assert(std::abs(sum_exp - 1.0f) < 1e-6f && "Sum of exp(log_softmax) should be 1");
    }
    
    // 2. All values should be negative or zero (since log(p) <= 0 for p <= 1)
    for (int i = 0; i < 6; i++) {
        assert(result.get_data_ptr()[i] <= 0.0f && "Log-softmax values should be <= 0");
    }
    
    // Test case 2: Edge case with identical values in a row
    Tensor<float> identical({1, 3});
    identical.get_data_ptr()[0] = 2.0f;
    identical.get_data_ptr()[1] = 2.0f;
    identical.get_data_ptr()[2] = 2.0f;
    
    Tensor<float> identical_result = identical.log_softmax_axis1();
    std::cout << "Identical values log-softmax:\n" << identical_result.to_string() << std::endl;
    
    // Should be log(1/3) ≈ -1.0986 for all elements
    float expected = std::log(1.0f / 3.0f);
    for (int i = 0; i < 3; i++) {
        assert(std::abs(identical_result.get_data_ptr()[i] - expected) < 1e-5f && 
               "Identical values should give log(1/n)");
    }
    
    // Test case 3: Test numerical stability with large values
    Tensor<float> large({1, 3});
    large.get_data_ptr()[0] = 1000.0f;
    large.get_data_ptr()[1] = 1001.0f;
    large.get_data_ptr()[2] = 1002.0f;
    
    Tensor<float> large_result = large.log_softmax_axis1();
    std::cout << "Large values log-softmax:\n" << large_result.to_string() << std::endl;
    
    // Should not have NaN or inf values
    for (int i = 0; i < 3; i++) {
        assert(!std::isnan(large_result.get_data_ptr()[i]) && "Should not produce NaN");
        assert(!std::isinf(large_result.get_data_ptr()[i]) && "Should not produce inf");
    }
    
    // Test case 4: Error case - should throw for non-2D tensor
    try {
        Tensor<float> wrong_dim({3});
        wrong_dim.log_softmax_axis1();
        assert(false && "Should throw for 1D tensor");
    } catch (const std::invalid_argument& e) {
        std::cout << "Correctly caught exception for 1D tensor: " << e.what() << std::endl;
    }
    
    std::cout << "All log_softmax_axis1 tests passed!" << std::endl;
    return true;
}

int main() {
    test_tensor_creation();
    assert(test_tensor_matmul());
    assert(test_tensor_sum());
    assert(test_dataloader());
    assert(test_model_init());
    assert(test_model_backward());
    assert(test_tensor_log_softmax_axis1());
}
