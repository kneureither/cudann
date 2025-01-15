#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "model.h"


int main() {

    Tensor<float> a;
    a.zeros({2, 3});
    std::cout << a.to_string() << std::endl;
    return 0;
}