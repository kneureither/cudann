#include "tensor.cuh"
#include "utils.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Helper function to print test results
void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// Helper function to compare floating point values
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Test tensor constructors and basic properties
bool test_tensor_constructors() {
    std::cout << "\n=== Testing Tensor Constructors ===" << std::endl;
    
    // Default constructor
    Tensor<float> t1;
    std::cout << "Default constructor shape: " << t1.shape_to_string() << std::endl;
    
    // Shape constructor
    Tensor<float> t2({3, 4});
    std::cout << "Shape constructor (3,4): " << t2.shape_to_string() << std::endl;
    std::cout << "Size: " << t2.get_size() << std::endl;
    
    // Device constructor
    Tensor<float> t3({2, 3}, Device::GPU);
    std::cout << "GPU device constructor (2,3): " << t3.shape_to_string() << std::endl;
    
    // Copy constructor
    Tensor<float> t4(t2);
    std::cout << "Copy constructor: " << t4.shape_to_string() << std::endl;
    
    return t2.get_size() == 12 && t2.get_shape()[0] == 3 && t2.get_shape()[1] == 4;
}

// Test tensor initialization methods
bool test_tensor_initialization() {
    std::cout << "\n=== Testing Tensor Initialization ===" << std::endl;
    
    // Test zeros
    Tensor<float> zeros_tensor({2, 3});
    zeros_tensor.zeros();
    std::cout << "Zeros tensor (2,3):\n" << zeros_tensor.to_string() << std::endl;
    
    // Verify all elements are zero
    bool all_zeros = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (zeros_tensor(i, j) != 0.0f) {
                all_zeros = false;
                break;
            }
        }
    }
    
    // Test ones
    Tensor<float> ones_tensor({2, 3});
    ones_tensor.ones();
    std::cout << "Ones tensor (2,3):\n" << ones_tensor.to_string() << std::endl;
    
    // Verify all elements are one
    bool all_ones = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (ones_tensor(i, j) != 1.0f) {
                all_ones = false;
                break;
            }
        }
    }
    
    // Test random uniform
    Tensor<float> random_tensor({2, 3});
    random_tensor.random_uniform(0.0f, 1.0f);
    std::cout << "Random uniform tensor (0,1):\n" << random_tensor.to_string() << std::endl;
    
    return all_zeros && all_ones;
}

// Test tensor access operators
bool test_tensor_access() {
    std::cout << "\n=== Testing Tensor Access ===" << std::endl;
    
    // Test 1D access
    Tensor<float> vec({5});
    vec.zeros();
    for (int i = 0; i < 5; i++) {
        vec.update_value_at_idx(static_cast<float>(i + 1), i);
    }
    std::cout << "1D tensor (manual fill):\n" << vec.to_string() << std::endl;
    
    // Test 2D access
    Tensor<float> mat({3, 3});
    mat.zeros();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat.update_value_at_idx(static_cast<float>(i * 3 + j + 1), i * 3 + j);
        }
    }
    std::cout << "2D tensor (manual fill):\n" << mat.to_string() << std::endl;
    
    // Verify values
    bool correct_values = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (mat(i, j) != static_cast<float>(i * 3 + j + 1)) {
                correct_values = false;
                break;
            }
        }
    }
    
    return vec[2] == 3.0f && mat(1, 1) == 5.0f && correct_values;
}

// Test update_value_at_idx
bool test_tensor_update_value() {
    std::cout << "\n=== Testing Tensor Update Value ===" << std::endl;
    
    Tensor<float> tensor({3});
    tensor.zeros();
    std::cout << "Before update:\n" << tensor.to_string() << std::endl;
    
    tensor.update_value_at_idx(42.0f, 1);
    std::cout << "After updating index 1 to 42:\n" << tensor.to_string() << std::endl;
    
    return float_equal(tensor[1], 42.0f);
}

// Test tensor addition
bool test_tensor_addition() {
    std::cout << "\n=== Testing Tensor Addition ===" << std::endl;
    
    // Test tensor + tensor (same shape)
    Tensor<float> a({2, 3});
    Tensor<float> b({2, 3});
    
    // Fill with simple values
    for (int i = 0; i < 6; i++) {
        a.update_value_at_idx(static_cast<float>(i + 1), i);
        b.update_value_at_idx(static_cast<float>(i + 1), i);
    }
    
    std::cout << "Tensor A:\n" << a.to_string() << std::endl;
    std::cout << "Tensor B:\n" << b.to_string() << std::endl;
    
    a += b;
    std::cout << "A += B result:\n" << a.to_string() << std::endl;
    
    // Test matrix + vector broadcasting
    Tensor<float> matrix({2, 3});
    Tensor<float> vector({3});
    
    for (int i = 0; i < 6; i++) {
        matrix.update_value_at_idx(static_cast<float>(i + 1), i);
    }
    for (int i = 0; i < 3; i++) {
        vector.update_value_at_idx(static_cast<float>(10), i);
    }
    
    std::cout << "Matrix before broadcast addition:\n" << matrix.to_string() << std::endl;
    std::cout << "Vector to add:\n" << vector.to_string() << std::endl;
    
    matrix += vector;
    std::cout << "Matrix + Vector result:\n" << matrix.to_string() << std::endl;
    
    return float_equal(a(0, 0), 2.0f) && float_equal(matrix(0, 0), 11.0f);
}

// Test tensor subtraction
bool test_tensor_subtraction() {
    std::cout << "\n=== Testing Tensor Subtraction ===" << std::endl;
    
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    
    // A = [[5, 6], [7, 8]]
    a.update_value_at_idx(5, 0); a.update_value_at_idx(6, 1);
    a.update_value_at_idx(7, 2); a.update_value_at_idx(8, 3);

    // B = [[1, 2], [3, 4]]
    b.update_value_at_idx(1, 0); b.update_value_at_idx(2, 1);
    b.update_value_at_idx(3, 2); b.update_value_at_idx(4, 3);
    
    std::cout << "Tensor A:\n" << a.to_string() << std::endl;
    std::cout << "Tensor B:\n" << b.to_string() << std::endl;
    
    a -= b;
    std::cout << "A -= B result:\n" << a.to_string() << std::endl;
    
    return float_equal(a(0, 0), 4.0f) && float_equal(a(1, 1), 4.0f);
}

// Test tensor multiplication
bool test_tensor_multiplication() {
    std::cout << "\n=== Testing Tensor Multiplication ===" << std::endl;
    
    // Element-wise multiplication
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    
    // A = [[2, 3], [4, 5]]
    a.update_value_at_idx(2, 0); a.update_value_at_idx(3, 1);
    a.update_value_at_idx(4, 2)4; a.update_value_at_idx(5, 3);

    // B = [[1, 2], [3, 4]]
    b.update_value_at_idx(1, 0); b.update_value_at_idx(2, 1);
    b.update_value_at_idx(3, 2); b.update_value_at_idx(4, 3);

    std::cout << "Tensor A:\n" << a.to_string() << std::endl;
    std::cout << "Tensor B:\n" << b.to_string() << std::endl;
    
    a *= b;
    std::cout << "A *= B (element-wise) result:\n" << a.to_string() << std::endl;
    
    // Scalar multiplication
    Tensor<float> c({2, 2});
    c.update_value_at_idx(1, 0); c.update_value_at_idx(2, 1);
    c.update_value_at_idx(3, 2); c.update_value_at_idx(4, 3);

    std::cout << "Tensor C before scalar multiplication:\n" << c.to_string() << std::endl;
    c *= 3.0f;
    std::cout << "C *= 3.0 result:\n" << c.to_string() << std::endl;
    
    return float_equal(a(0, 0), 2.0f) && float_equal(c(0, 1), 6.0f);
}

// Test greater than operator
bool test_tensor_greater_than() {
    std::cout << "\n=== Testing Tensor Greater Than ===" << std::endl;
    
    Tensor<float> a({2, 3});
    // Fill with values [1, 2, 3, 4, 5, 6]
    for (int i = 0; i < 6; i++) {
        a.update_value_at_idx(static_cast<float>(i + 1), i);
    }
    
    std::cout << "Input tensor:\n" << a.to_string() << std::endl;
    
    Tensor<float> result = a > 3.5f;
    std::cout << "Result of > 3.5:\n" << result.to_string() << std::endl;
    
    return float_equal(result(0, 0), 0.0f) && float_equal(result(1, 1), 1.0f);
}

// Test matrix multiplication
bool test_tensor_matmul() {
    std::cout << "\n=== Testing Matrix Multiplication ===" << std::endl;
    
    // A: 2x3 matrix
    Tensor<float> a({2, 3});
    // A = [[1, 2, 3], [4, 5, 6]]
    a.update_value_at_idx(1, 0); a.update_value_at_idx(2, 1); a.update_value_at_idx(3, 2);
    a.update_value_at_idx(4, 3); a.update_value_at_idx(5, 4); a.update_value_at_idx(6, 5);

    // B: 3x2 matrix
    Tensor<float> b({3, 2});
    // B = [[1, 2], [3, 4], [5, 6]]
    b.update_value_at_idx(1, 0); b.update_value_at_idx(2, 1);
    b.update_value_at_idx(3, 2); b.update_value_at_idx(4, 3);
    b.update_value_at_idx(5, 4); b.update_value_at_idx(6, 5);

    std::cout << "Matrix A (2x3):\n" << a.to_string() << std::endl;
    std::cout << "Matrix B (3x2):\n" << b.to_string() << std::endl;
    
    Tensor<float> c = a.matmul(b);
    std::cout << "A @ B result (2x2):\n" << c.to_string() << std::endl;
    
    // Expected: [[22, 28], [49, 64]]
    // Row 0: [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    // Row 1: [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [49, 64]
    
    return float_equal(c(0, 0), 22.0f) && float_equal(c(0, 1), 28.0f) && 
           float_equal(c(1, 0), 49.0f) && float_equal(c(1, 1), 64.0f);
}

// Test matrix transpose
bool test_tensor_transpose() {
    std::cout << "\n=== Testing Matrix Transpose ===" << std::endl;
    
    Tensor<float> a({2, 3});
    // A = [[1, 2, 3], [4, 5, 6]]
    a.update_value_at_idx(1, 0); a.update_value_at_idx(2, 1); a.update_value_at_idx(3, 2);
    a.update_value_at_idx(4, 3); a.update_value_at_idx(5, 4); a.update_value_at_idx(6, 5);

    std::cout << "Original matrix (2x3):\n" << a.to_string() << std::endl;
    
    Tensor<float> a_t = a.transpose();
    std::cout << "Transposed matrix (3x2):\n" << a_t.to_string() << std::endl;
    
    // Expected: [[1, 4], [2, 5], [3, 6]]
    return float_equal(a_t(0, 0), 1.0f) && float_equal(a_t(0, 1), 4.0f) &&
           float_equal(a_t(1, 0), 2.0f) && float_equal(a_t(2, 1), 6.0f) &&
           a_t.get_shape()[0] == 3 && a_t.get_shape()[1] == 2;
}

// Test exponential function
bool test_tensor_exp() {
    std::cout << "\n=== Testing Tensor Exponential ===" << std::endl;
    
    Tensor<float> a({2, 2});
    // A = [[0, 1], [2, 3]]
    a.update_value_at_idx(0, 0); a.update_value_at_idx(1, 1);
    a.update_value_at_idx(2, 2); a.update_value_at_idx(3, 3);

    std::cout << "Input tensor:\n" << a.to_string() << std::endl;
    
    Tensor<float> exp_result = a.exp();
    std::cout << "Exponential result:\n" << exp_result.to_string() << std::endl;
    
    // Expected: [[1, e], [e^2, e^3]]
    return float_equal(exp_result(0, 0), 1.0f, 1e-4f) && 
           float_equal(exp_result(0, 1), std::exp(1.0f), 1e-4f) &&
           float_equal(exp_result(1, 0), std::exp(2.0f), 1e-4f);
}

// Test logarithm function
bool test_tensor_log() {
    std::cout << "\n=== Testing Tensor Logarithm ===" << std::endl;
    
    Tensor<float> a({2, 2});
    // A = [[1, e], [e^2, 10]]
    a.update_value_at_idx(1.0f, 0); a.update_value_at_idx(std::exp(1.0f), 1);
    a.update_value_at_idx(std::exp(2.0f), 2); a.update_value_at_idx(10.0f, 3);

    std::cout << "Input tensor:\n" << a.to_string() << std::endl;
    
    Tensor<float> log_result = a.log();
    std::cout << "Logarithm result:\n" << log_result.to_string() << std::endl;
    
    // Expected: [[0, 1], [2, ln(10)]]
    return float_equal(log_result(0, 0), 0.0f, 1e-4f) && 
           float_equal(log_result(0, 1), 1.0f, 1e-4f) &&
           float_equal(log_result(1, 0), 2.0f, 1e-4f);
}

// Test sum operations
bool test_tensor_sum() {
    std::cout << "\n=== Testing Tensor Sum ===" << std::endl;
    
    Tensor<float> a({3, 4});
    // Fill with values 1-12
    for (int i = 0; i < 12; i++) {
        a.update_value_at_idx(static_cast<float>(i + 1), i);
    }
    
    std::cout << "Input tensor (3x4):\n" << a.to_string() << std::endl;
    
    // Sum along axis 0 (columns)
    Tensor<float> sum_axis0 = a.sum(0);
    std::cout << "Sum along axis 0 (shape should be [4]):\n" << sum_axis0.to_string() << std::endl;
    
    // Sum along axis 1 (rows)
    Tensor<float> sum_axis1 = a.sum(1);
    std::cout << "Sum along axis 1 (shape should be [3]):\n" << sum_axis1.to_string() << std::endl;
    
    // Expected axis 0: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
    // Expected axis 1: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
    
    return float_equal(sum_axis0[0], 15.0f) && float_equal(sum_axis0[3], 24.0f) &&
           float_equal(sum_axis1[0], 10.0f) && float_equal(sum_axis1[2], 42.0f);
}

// Test argmax operations
bool test_tensor_argmax() {
    std::cout << "\n=== Testing Tensor Argmax ===" << std::endl;
    
    Tensor<float> a({2, 4});
    // A = [[1, 4, 2, 3], [8, 5, 7, 6]]
    a.update_value_at_idx(1, 0); a.update_value_at_idx(4, 1); a.update_value_at_idx(2, 2); a.update_value_at_idx(3, 3);
    a.update_value_at_idx(8, 4); a.update_value_at_idx(5, 5); a.update_value_at_idx(7, 6); a.update_value_at_idx(6, 7);

    std::cout << "Input tensor (2x4):\n" << a.to_string() << std::endl;
    
    // Argmax along axis 0
    Tensor<int> argmax_axis0 = a.argmax(0);
    std::cout << "Argmax along axis 0:\n" << argmax_axis0.to_string() << std::endl;
    
    // Argmax along axis 1
    Tensor<int> argmax_axis1 = a.argmax(1);
    std::cout << "Argmax along axis 1:\n" << argmax_axis1.to_string() << std::endl;
    
    // Expected axis 0: [1, 1, 1, 1] (second row has max in each column)
    // Expected axis 1: [1, 0] (index 1 for first row, index 0 for second row)
    
    return argmax_axis0[0] == 1 && argmax_axis0[1] == 1 &&
           argmax_axis1[0] == 1 && argmax_axis1[1] == 0;
}

// Test softmax
bool test_tensor_softmax() {
    std::cout << "\n=== Testing Tensor Softmax ===" << std::endl;
    
    Tensor<float> a({2, 3});
    // A = [[1, 2, 3], [4, 5, 6]]
    a.update_value_at_idx(1, 0); a.update_value_at_idx(2, 1); a.update_value_at_idx(3, 2);
    a.update_value_at_idx(4, 3); a.update_value_at_idx(5, 4); a.update_value_at_idx(6, 5);

    std::cout << "Input tensor:\n" << a.to_string() << std::endl;
    
    Tensor<float> softmax_result = a.softmax_axis1();
    std::cout << "Softmax result:\n" << softmax_result.to_string() << std::endl;
    
    // Verify that each row sums to 1
    float row0_sum = softmax_result(0, 0) + softmax_result(0, 1) + softmax_result(0, 2);
    float row1_sum = softmax_result(1, 0) + softmax_result(1, 1) + softmax_result(1, 2);
    
    std::cout << "Row 0 sum: " << row0_sum << std::endl;
    std::cout << "Row 1 sum: " << row1_sum << std::endl;
    
    return float_equal(row0_sum, 1.0f, 1e-4f) && float_equal(row1_sum, 1.0f, 1e-4f);
}

// Test log softmax
bool test_tensor_log_softmax() {
    std::cout << "\n=== Testing Tensor Log Softmax ===" << std::endl;
    
    Tensor<float> a({2, 3});
    // A = [[1, 2, 3], [4, 5, 6]]
    a.update_value_at_idx(1, 0); a.update_value_at_idx(2, 1); a.update_value_at_idx(3, 2);
    a.update_value_at_idx(4, 3); a.update_value_at_idx(5, 4); a.update_value_at_idx(6, 5);

    std::cout << "Input tensor:\n" << a.to_string() << std::endl;
    
    Tensor<float> log_softmax_result = a.log_softmax_axis1();
    std::cout << "Log-softmax result:\n" << log_softmax_result.to_string() << std::endl;
    
    // Verify that exp(log_softmax) sums to 1 for each row
    float row0_sum = std::exp(log_softmax_result(0, 0)) + std::exp(log_softmax_result(0, 1)) + std::exp(log_softmax_result(0, 2));
    float row1_sum = std::exp(log_softmax_result(1, 0)) + std::exp(log_softmax_result(1, 1)) + std::exp(log_softmax_result(1, 2));
    
    std::cout << "Exp(row 0) sum: " << row0_sum << std::endl;
    std::cout << "Exp(row 1) sum: " << row1_sum << std::endl;
    
    // All log-softmax values should be <= 0
    bool all_negative = true;
    for (int i = 0; i < 6; i++) {
        if (log_softmax_result.get_data_ptr()[i] > 0.0f) {
            all_negative = false;
            break;
        }
    }
    
    return float_equal(row0_sum, 1.0f, 1e-4f) && float_equal(row1_sum, 1.0f, 1e-4f) && all_negative;
}

// Test assignment operator
bool test_tensor_assignment() {
    std::cout << "\n=== Testing Tensor Assignment ===" << std::endl;
    
    Tensor<float> a({2, 2});
    a.update_value_at_idx(1, 0); a.update_value_at_idx(2, 1);
    a.update_value_at_idx(3, 2); a.update_value_at_idx(4, 3);
    
    std::cout << "Original tensor A:\n" << a.to_string() << std::endl;
    
    Tensor<float> b;
    b = a;  // Assignment
    
    std::cout << "Assigned tensor B:\n" << b.to_string() << std::endl;
    
    // Modify original to ensure deep copy
    a.get_data_ptr()[0] = 999;
    std::cout << "After modifying A[0] to 999:\n";
    std::cout << "Tensor A:\n" << a.to_string() << std::endl;
    std::cout << "Tensor B:\n" << b.to_string() << std::endl;
    
    return float_equal(b(0, 0), 1.0f) && float_equal(a(0, 0), 999.0f);
}

// Test device management
bool test_device_management() {
    std::cout << "\n=== Testing Device Management ===" << std::endl;
    
    Tensor<float> gpu_tensor({2, 2}, Device::GPU);
    std::cout << "GPU tensor device: " << (gpu_tensor.get_device() == Device::GPU ? "GPU" : "CPU") << std::endl;
    
    // Note: CPU device functionality might not be fully implemented
    // This test mainly checks that the interface works
    return gpu_tensor.get_device() == Device::GPU;
}

// Main test runner
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "       CUDA TENSOR COMPREHENSIVE TESTS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_passed = true;
    
    // Run all tests
    all_passed &= test_tensor_constructors();
    print_test_result("Tensor Constructors", test_tensor_constructors());
    
    all_passed &= test_tensor_initialization();
    print_test_result("Tensor Initialization", test_tensor_initialization());
    
    all_passed &= test_tensor_access();
    print_test_result("Tensor Access", test_tensor_access());
    
    all_passed &= test_tensor_update_value();
    print_test_result("Tensor Update Value", test_tensor_update_value());
    
    all_passed &= test_tensor_addition();
    print_test_result("Tensor Addition", test_tensor_addition());
    
    all_passed &= test_tensor_subtraction();
    print_test_result("Tensor Subtraction", test_tensor_subtraction());
    
    all_passed &= test_tensor_multiplication();
    print_test_result("Tensor Multiplication", test_tensor_multiplication());
    
    all_passed &= test_tensor_greater_than();
    print_test_result("Tensor Greater Than", test_tensor_greater_than());
    
    all_passed &= test_tensor_matmul();
    print_test_result("Matrix Multiplication", test_tensor_matmul());
    
    all_passed &= test_tensor_transpose();
    print_test_result("Matrix Transpose", test_tensor_transpose());
    
    all_passed &= test_tensor_exp();
    print_test_result("Tensor Exponential", test_tensor_exp());
    
    all_passed &= test_tensor_log();
    print_test_result("Tensor Logarithm", test_tensor_log());
    
    all_passed &= test_tensor_sum();
    print_test_result("Tensor Sum", test_tensor_sum());
    
    all_passed &= test_tensor_argmax();
    print_test_result("Tensor Argmax", test_tensor_argmax());
    
    all_passed &= test_tensor_softmax();
    print_test_result("Tensor Softmax", test_tensor_softmax());
    
    all_passed &= test_tensor_log_softmax();
    print_test_result("Tensor Log Softmax", test_tensor_log_softmax());
    
    all_passed &= test_tensor_assignment();
    print_test_result("Tensor Assignment", test_tensor_assignment());
    
    all_passed &= test_device_management();
    print_test_result("Device Management", test_device_management());
    
    // Final results
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED ❌" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return all_passed ? 0 : 1;
}
