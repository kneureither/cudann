#include "dataset.h"
#include "dataloader.h"
#include "tensor.cuh"

#include <iostream>
#include <bit>
#include <fstream>
#include <random>
#include <algorithm>

mnist_image_t *DataLoader::get_sample(int idx)
{
    return &(this->dataset.images[idx]);
}

uint8_t DataLoader::get_label(int idx)
{
    return this->dataset.labels[idx];
}

void DataLoader::_load_images_from_file()
{
    this->dataset.images = nullptr;

    std::basic_ifstream<char> file(this->data_path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << this->data_path << '\n';
        return;
    }

    mnist_image_file_header_t header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Could not read image file header from: " << this->data_path << '\n';
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
        std::cerr << "Invalid header read from image file: " << this->data_path
                  << " (" << std::hex << header.magic_number << " not " << MNIST_IMAGE_MAGIC << ")\n";
        return;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows)
    {
        std::cerr << "Invalid number of image rows in image file " << this->data_path
                  << " (" << header.number_of_rows << " not " << MNIST_IMAGE_WIDTH << ")\n";
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns)
    {
        std::cerr << "Invalid number of image columns in image file " << this->data_path
                  << " (" << header.number_of_columns << " not " << MNIST_IMAGE_HEIGHT << ")\n";
    }

    this->dataset.size = header.number_of_images;
    this->dataset.images = new mnist_image_t[this->dataset.size];

    if (!file.read(reinterpret_cast<char *>(this->dataset.images), this->dataset.size * sizeof(mnist_image_t)))
    {
        std::cerr << "Could not read " << this->dataset.size << " images from: " << this->data_path << '\n';
        delete[] this->dataset.images;
        this->dataset.images = nullptr;
        return;
    }

    logger("Loaded " + std::to_string(this->dataset.size) + " images from: " + this->data_path, "INFO", __FILE__, __LINE__);
}

void DataLoader::_load_labels_from_file()
{
    this->dataset.labels = nullptr;

    std::basic_ifstream<char> file(this->label_path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << this->label_path << '\n';
        return;
    }

    mnist_label_file_header_t header;
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Could not read label file header from: " << this->label_path << '\n';
        return;
    }

// Convert from big endian to little endian
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    header.magic_number = __builtin_bswap32(header.magic_number);
    header.number_of_labels = __builtin_bswap32(header.number_of_labels);
#endif

    if (MNIST_LABEL_MAGIC != header.magic_number)
    {
        std::cout << "Invalid header read from label file: " << this->label_path
                  << " (" << header.magic_number << " not " << MNIST_LABEL_MAGIC << ")\n";
        return;
    }

    this->dataset.size = header.number_of_labels;
    this->dataset.labels = new uint8_t[this->dataset.size];

    if (!file.read(reinterpret_cast<char *>(this->dataset.labels), this->dataset.size))
    {
        std::cerr << "Could not read " << this->dataset.size << " labels from: " << this->label_path << '\n';
        delete[] this->dataset.labels;
        this->dataset.labels = nullptr;
        return;
    }

    logger("Loaded " + std::to_string(this->dataset.size) + " labels from: " + this->label_path, "INFO", __FILE__, __LINE__);
}

void DataLoader::_free_dataset()
{
    if (this->dataset.images != nullptr)
        delete[] this->dataset.images;
    if (this->dataset.labels != nullptr)
        delete[] this->dataset.labels;
}

void DataLoader::shuffle_indices()
{
    // Shuffle the indices
    this->indices = std::vector<int>(this->get_size());
    for (int i = 0; i < this->get_size(); i++)
        this->indices[i] = i;

    if (this->shuffle)
    {
        std::shuffle(this->indices.begin(), this->indices.end(), std::default_random_engine(this->shuffle_seed));
    }
}

size_t DataLoader::get_size()
{
    return this->dataset.size;
}

size_t DataLoader::get_max_num_batches()
{
    return std::floor(this->get_size() / this->batch_size);
}

Tensor<float> DataLoader::load_data_batch(int batch_idx)
{
    Tensor<float> data({(size_t)this->batch_size, MNIST_IMAGE_SIZE});
    
    // Create CPU buffer to prepare data efficiently
    std::vector<float> cpu_data(this->batch_size * MNIST_IMAGE_SIZE);
    
    for (int i = 0; i < this->batch_size; i++)
    {
        mnist_image_t *image = this->get_sample(this->indices[batch_idx * this->batch_size + i]);
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++)
        {
            cpu_data[i * MNIST_IMAGE_SIZE + j] = (float)image->pixels[j] / 255.0f;
        }
    }
    
    // Copy data directly to GPU memory
    CUDA_CHECK(cudaMemcpy(data.get_data_ptr(), cpu_data.data(), 
                         cpu_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    return data;
}

Tensor<int> DataLoader::load_labels_batch(int batch_idx)
{
    Tensor<int> labels({(size_t)this->batch_size});
    
    // Create CPU buffer to prepare data efficiently
    std::vector<int> cpu_labels(this->batch_size);
    
    for (int i = 0; i < this->batch_size; i++)
    {
        cpu_labels[i] = (int)this->get_label(this->indices[batch_idx * this->batch_size + i]);
    }
    
    // Copy data directly to GPU memory
    CUDA_CHECK(cudaMemcpy(labels.get_data_ptr(), cpu_labels.data(), 
                         cpu_labels.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    return labels;
}

Tensor<float> DataLoader::load_data()
{
    Tensor<float> data({(size_t)this->get_size(), MNIST_IMAGE_SIZE});

    // Create CPU buffer to prepare data efficiently
    std::vector<float> cpu_data(this->get_size() * MNIST_IMAGE_SIZE);

    // Load the data
    for (int i = 0; i < this->get_size(); i++)
    {
        mnist_image_t *image = this->get_sample(this->indices[i]);
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++)
        {
            cpu_data[i * MNIST_IMAGE_SIZE + j] = (float)image->pixels[j] / 255.0f;
        }
    }
    
    // Copy data directly to GPU memory
    CUDA_CHECK(cudaMemcpy(data.get_data_ptr(), cpu_data.data(), 
                         cpu_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    return data;
};

Tensor<int> DataLoader::load_labels()
{
    Tensor<int> labels({(size_t) this->get_size()});

    // Create CPU buffer to prepare data efficiently
    std::vector<int> cpu_labels(this->get_size());

    for (int i = 0; i < this->get_size(); i++)
    {
        cpu_labels[i] = (int)this->get_label(this->indices[i]);
    }
    
    // Copy data directly to GPU memory
    CUDA_CHECK(cudaMemcpy(labels.get_data_ptr(), cpu_labels.data(), 
                         cpu_labels.size() * sizeof(int), cudaMemcpyHostToDevice));

    return labels;
};

std::string DataLoader::get_image_as_string(int idx)
{
    mnist_image_t *image = this->get_sample(idx);
    std::stringstream ss;
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++)
    {
        int pixel = (int)image->pixels[i];
        if (pixel == 0)
        {
            ss << "  0 ";
        }
        else if (pixel > 0 && pixel < 10)
        {
            ss << "  " << pixel << " ";
        }
        else if (pixel >= 10 && pixel < 100)
        {
            ss << " " << pixel << " ";
        }
        else if (pixel >= 100 && pixel < 255)
        {
            ss << "" << pixel << " ";
        }
    }
    return ss.str();
};
