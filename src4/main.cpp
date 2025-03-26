/*Training Script for the model*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "model.h"
#include "dataloader.h"
#include "tensor.h"


int main() {
    Tensor<float> a({2});
    a.zeros();
    std::cout << "1-dim tensor: \n" << a.to_string() << std::endl;

    Tensor<float> b({2, 3});
    b.zeros();
    std::cout << "2-dim tensor: \n" << b.to_string() << std::endl;

    Tensor<float> c({2, 3, 4});
    c.zeros();
    std::cout << "3-dim tensor: \n" << c.to_string() << std::endl;

    Tensor<float> d({2, 3, 4, 5});
    d.zeros();
    std::cout << "4-dim tensor: \n" << d.to_string() << std::endl;
    
    return 0;
}