#include "dataset.h"

#include <iostream>
#include <bit>
#include <fstream>

Dataset::Dataset()
{
    this->_read_from_file();
}

Dataset::~Dataset()
{
    this->_free_dataset(&(this->train_dataset));
    this->_free_dataset(&(this->test_dataset));
}

void Dataset::_read_from_file()
{
    std::cout << "Reading dataset..." << std::endl;
    // read images
    this->_get_images_from_file(this->test_images_file, &(this->test_dataset));
    this->_get_images_from_file(this->train_images_file, &(this->train_dataset));

    // read labels
    this->_get_labels_from_file(this->test_labels_file, &(this->test_dataset));
    this->_get_labels_from_file(this->train_labels_file, &(this->train_dataset));
}

int Dataset::get_train_size()
{
    return this->train_dataset.size;
}

int Dataset::get_test_size()
{
    return this->test_dataset.size;
}

void Dataset::_free_dataset(mnist_dataset_t *dataset)
{
    if (dataset->images != NULL)
        delete[] dataset->images;
    if (dataset->labels != NULL)
        delete[] dataset->labels;
}

mnist_image_t *Dataset::get_train_sample(int index)
{
    return &(this->train_dataset.images[index]);
}

mnist_image_t *Dataset::get_test_sample(int index)
{
    return &(this->test_dataset.images[index]);
}

uint8_t Dataset::get_train_label(int index)
{
    return this->train_dataset.labels[index];
}

uint8_t Dataset::get_test_label(int index)
{
    return this->test_dataset.labels[index];
}

/**
 * Read labels from file.
 *
 * File format: http://yann.lecun.com/exdb/mnist/
 */
void Dataset::_get_labels_from_file(const char *path, mnist_dataset_t *dataset)
{
    dataset->labels = nullptr;

    std::basic_ifstream<char> file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << path << '\n';
        return;
    }

    mnist_label_file_header_t header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Could not read label file header from: " << path << '\n';
        return;
    }

// Convert from big endian to little endian
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    header.magic_number = __builtin_bswap32(header.magic_number);
    header.number_of_labels = __builtin_bswap32(header.number_of_labels);
#endif

    if (MNIST_LABEL_MAGIC != header.magic_number)
    {
        std::cout << "Invalid header read from label file: " << path
                  << " (" << header.magic_number << " not " << MNIST_LABEL_MAGIC << ")\n";
        return;
    }

    dataset->size = header.number_of_labels;
    dataset->labels = new uint8_t[dataset->size];

    if (!file.read(reinterpret_cast<char *>(dataset->labels), dataset->size))
    {
        std::cerr << "Could not read " << dataset->size << " labels from: " << path << '\n';
        delete[] dataset->labels;
        dataset->labels = nullptr;
        return;
    }
}

void Dataset::_get_images_from_file(const char *path, mnist_dataset_t *dataset)
{
    dataset->images = nullptr;

    std::basic_ifstream<char> file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << path << '\n';
        return;
    }

    mnist_image_file_header_t header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Could not read image file header from: " << path << '\n';
        return;
    }

// Convert from big endian to little endian
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    header.magic_number = __builtin_bswap32(header.magic_number);
    header.number_of_images = __builtin_bswap32(header.number_of_images);
    header.number_of_rows = __builtin_bswap32(header.number_of_rows);
    header.number_of_columns = __builtin_bswap32(header.number_of_columns);
#endif

    if (MNIST_IMAGE_MAGIC != header.magic_number)
    {
        std::cerr << "Invalid header read from image file: " << path
                  << " (" << std::hex << header.magic_number << " not " << MNIST_IMAGE_MAGIC << ")\n";
        return;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows)
    {
        std::cerr << "Invalid number of image rows in image file " << path
                  << " (" << header.number_of_rows << " not " << MNIST_IMAGE_WIDTH << ")\n";
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns)
    {
        std::cerr << "Invalid number of image columns in image file " << path
                  << " (" << header.number_of_columns << " not " << MNIST_IMAGE_HEIGHT << ")\n";
    }

    dataset->size = header.number_of_images;
    dataset->images = new mnist_image_t[dataset->size];

    if (!file.read(reinterpret_cast<char *>(dataset->images), dataset->size * sizeof(mnist_image_t)))
    {
        std::cerr << "Could not read " << dataset->size << " images from: " << path << '\n';
        delete[] dataset->images;
        dataset->images = nullptr;
        return;
    }
}