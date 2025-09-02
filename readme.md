# Simple CUDA NN MNist Classifier


This repository contains a C++ implementation of a simple neural net, that runs on CPU and can also be build for cuda. It can learn to classify digits of the Mnist handwritten digits data set. In its simplest form, the classifier consists of a single layer fully connected NN with softmax cross entropy loss, but more layers can be added. For now only the ReLu activation function is implemented.

All cuda operations for model computations are contained in the tensor module, which header-only for CPU (`tensor.h`) and has another object file (`tensor.cuh` and `tensor.cu`) for the CUDA version. Also, the `srcu/dataloader.cpp` class uses a MemCopy operation. Aside from that, the tensor api allows for quick templated prototyping (for now `<int>` and `<float>` are supported) and generates code that can be run on CPU as well as on GPU, just by choosing the specific header to include.

The code was developed by Konstantin Neureither ([dev@kneureither.de](mailto:dev@kneureither.de)), please reach out for copyright and bug reports. Furthermore, the dataloader implementation was taken and adapted from Andrew Carters' [Mnist-Neural-Network-Plain-C](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c).

## Run the training
### Make the build

```
# Configure the build
cmake -B build -S .

# Build the project
cmake --build build
```

### Run CUDA
```
./build/cumain
```

### Run CPU 
```
./build/main
```

## To Do

There are two main tasks open
- There is a problem in the loss implementation with cuda (it still needs to be ported to a kernel)
- the `src4/dataloader.cpp` class, which handles the CPU mode, needs to be merged with the CUDA version `srcu/dataloader.cpp`
