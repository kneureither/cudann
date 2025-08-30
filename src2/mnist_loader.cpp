#include "neural_network.h"
#include "mnist_loader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
    struct ImageFileHeader {
        uint32_t magic_number;
        uint32_t num_images;
        uint32_t num_rows;
        uint32_t num_cols;
    } __attribute__((packed));

    struct LabelFileHeader {
        uint32_t magic_number;
        uint32_t num_labels;
    } __attribute__((packed));

    // Helper function to convert from big-endian to host endian
    template<typename T>
    void convert_endian(T& value) {
        #if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
            value = __builtin_bswap32(value);
        #endif
    }

    // Helper function to read and validate headers
    template <typename Header>
    Header read_header(std::ifstream &file, uint32_t expected_magic_number)
    {
        Header header;
        file.read(reinterpret_cast<char *>(&header), sizeof(Header));

        // Use temporary variables
        uint32_t magic = header.magic_number;
        convert_endian(magic);
        header.magic_number = magic;

        if (header.magic_number != expected_magic_number)
        {
            throw std::runtime_error("Invalid magic number");
        }

        if constexpr (std::is_same_v<Header, ImageFileHeader>)
        {
            uint32_t num_images = header.num_images;
            uint32_t num_rows = header.num_rows;
            uint32_t num_cols = header.num_cols;

            convert_endian(num_images);
            convert_endian(num_rows);
            convert_endian(num_cols);

            header.num_images = num_images;
            header.num_rows = num_rows;
            header.num_cols = num_cols;
        }
        else if constexpr (std::is_same_v<Header, LabelFileHeader>)
        {
            uint32_t num_labels = header.num_labels;
            convert_endian(num_labels);
            header.num_labels = num_labels;
        }

        return header;
    }
}

void MNISTLoader::load_mnist_data(const std::string& image_file, 
                                 const std::string& label_file,
                                 Tensor& images,
                                 Tensor& labels) {
    // Open image file
    std::ifstream image_stream(image_file, std::ios::binary);
    if (!image_stream) {
        throw std::runtime_error("Could not open file: " + image_file);
    }

    // Read and validate image header
    const auto img_header = read_header<ImageFileHeader>(image_stream, 0x00000803);
    
    // Open label file
    std::ifstream label_stream(label_file, std::ios::binary);
    if (!label_stream) {
        throw std::runtime_error("Could not open file: " + label_file);
    }

    // Read and validate label header
    const auto label_header = read_header<LabelFileHeader>(label_stream, 0x00000801);

    // Verify matching sizes
    if (label_header.num_labels != img_header.num_images) {
        throw std::runtime_error("Number of labels doesn't match number of images");
    }

    // Initialize tensors
    const size_t num_pixels = img_header.num_rows * img_header.num_cols;
    images = Tensor({img_header.num_images, num_pixels});
    labels = Tensor({label_header.num_labels, 10}); // 10 classes
    labels.zeros();
    std::cout << "num_pixels: " << num_pixels << " num_images: " << img_header.num_images << " num_labels: " << label_header.num_labels << std::endl;

    // Read image data
    std::vector<uint8_t> pixel_buffer(num_pixels);
    for (uint32_t i = 0; i < img_header.num_images; ++i) {
        if (!image_stream.read(reinterpret_cast<char*>(pixel_buffer.data()), 
                             num_pixels)) {
            throw std::runtime_error("Failed to read image data");
        }
        
        // Normalize and copy to tensor
        for (size_t j = 0; j < num_pixels; ++j) {
            images.data()[i * num_pixels + j] = pixel_buffer[j] / 255.0f;
        }
    }

    // Read and process labels
    std::vector<uint8_t> label_buffer(label_header.num_labels);
    if (!label_stream.read(reinterpret_cast<char*>(label_buffer.data()), 
                          label_header.num_labels)) {
        throw std::runtime_error("Failed to read label data");
    }

    // Convert to one-hot encoding
    for (uint32_t i = 0; i < label_header.num_labels; ++i) {
        labels.data()[i * 10 + label_buffer[i]] = 1.0f;
    }
}

void MNISTLoader::load_train_data(const std::string& image_file, 
                                 const std::string& label_file) {
    std::cout << "Loading train data" << std::endl;
    load_mnist_data(image_file, label_file, m_train_images, m_train_labels);
}

void MNISTLoader::load_test_data(const std::string& image_file, 
                                const std::string& label_file) {
    load_mnist_data(image_file, label_file, m_test_images, m_test_labels);
}

const Tensor& MNISTLoader::get_training_images() const {
    return m_train_images;
}

const Tensor& MNISTLoader::get_training_labels() const {
    return m_train_labels;
}

const Tensor& MNISTLoader::get_test_images() const {
    return m_test_images;
}

const Tensor& MNISTLoader::get_test_labels() const {
    return m_test_labels;
}
