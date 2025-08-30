#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "neural_network.h"

// MNIST Data Loader
class MNISTLoader
{
public:
    void load_train_data(const std::string &image_file,
                         const std::string &label_file);
    void load_test_data(const std::string &image_file,
                        const std::string &label_file);

    void load_mnist_data(const std::string &image_file,
                         const std::string &label_file,
                         Tensor &images,
                         Tensor &labels);
                    

    const Tensor& get_training_images() const;
    const Tensor& get_training_labels() const;
    const Tensor& get_test_images() const;
    const Tensor& get_test_labels() const;

private:
    Tensor m_train_images;
    Tensor m_train_labels;
    Tensor m_test_images;
    Tensor m_test_labels;
};

#endif // MNIST_LOADER_H