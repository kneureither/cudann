#include "dataset.h"

#include <iostream>
#include <bit>

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
    FILE *stream;
    mnist_label_file_header_t header;
    dataset->labels = NULL;

    stream = fopen(path, "rb");

    if (NULL == stream)
    {
        fprintf(stderr, "Could not open file: %s\n", path);
        return;
    }

    if (1 != fread(&header, sizeof(mnist_label_file_header_t), 1, stream))
    {
        fprintf(stderr, "Could not read label file header from: %s\n", path);
        fclose(stream);
        return;
    }

    // Convert from big endian to little endian
    if (std::endian::native == std::endian::little)
    {
        header.magic_number = __builtin_bswap32(header.magic_number);
        header.number_of_labels = __builtin_bswap32(header.number_of_labels);
    }

    if (MNIST_LABEL_MAGIC != header.magic_number)
    {
        std::cout << "Invalid header read from label file: " << path << " (" << header.magic_number << " not " << MNIST_LABEL_MAGIC << ")" << std::endl;
        fclose(stream);
        return;
    }

    dataset->size = header.number_of_labels;
    dataset->labels = new uint8_t[dataset->size];

    if (dataset->size != fread(dataset->labels, 1, dataset->size, stream))
    {
        fprintf(stderr, "Could not read %d labels from: %s\n", dataset->size, path);
        delete[] dataset->labels;
        dataset->labels = NULL;
        fclose(stream);
        return;
    }

    fclose(stream);
}

void Dataset::_get_images_from_file(const char *path, mnist_dataset_t *dataset)
{
    FILE *stream;
    mnist_image_file_header_t header;

    stream = fopen(path, "rb");

    dataset->images = NULL;
    if (NULL == stream)
    {
        fprintf(stderr, "Could not open file: %s\n", path);
        return;
    }

    if (1 != fread(&header, sizeof(mnist_image_file_header_t), 1, stream))
    {
        fprintf(stderr, "Could not read image file header from: %s\n", path);
        fclose(stream);
        return;
    }

    header.magic_number = __builtin_bswap32(header.magic_number);
    header.number_of_images = __builtin_bswap32(header.number_of_images);
    header.number_of_rows = __builtin_bswap32(header.number_of_rows);
    header.number_of_columns = __builtin_bswap32(header.number_of_columns);

    if (MNIST_IMAGE_MAGIC != header.magic_number)
    {
        fprintf(stderr, "Invalid header read from image file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_IMAGE_MAGIC);
        fclose(stream);
        return;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows)
    {
        fprintf(stderr, "Invalid number of image rows in image file %s (%d not %d)\n", path, header.number_of_rows, MNIST_IMAGE_WIDTH);
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns)
    {
        fprintf(stderr, "Invalid number of image columns in image file %s (%d not %d)\n", path, header.number_of_columns, MNIST_IMAGE_HEIGHT);
    }

    dataset->size = header.number_of_images;
    dataset->images = (mnist_image_t *)malloc(dataset->size * sizeof(mnist_image_t));

    if (dataset->images == NULL)
    {
        fprintf(stderr, "Could not allocated memory for %d images\n", dataset->size);
        fclose(stream);
        return;
    }

    if (dataset->size != fread(dataset->images, sizeof(mnist_image_t), dataset->size, stream))
    {
        fprintf(stderr, "Could not read %d images from: %s\n", dataset->size, path);
        delete[] dataset->images;
        dataset->images = NULL;
        fclose(stream);
        return;
    }

    fclose(stream);
}