#ifndef DATASET_H
#define DATASET_H

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH *MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

typedef struct mnist_label_file_header_t_
{
    uint32_t magic_number;
    uint32_t number_of_labels;
} __attribute__((packed)) mnist_label_file_header_t;

typedef struct mnist_image_file_header_t_
{
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} __attribute__((packed)) mnist_image_file_header_t;

typedef struct mnist_image_t_
{
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image_t;

typedef struct mnist_dataset_t_
{
    mnist_image_t *images;
    uint8_t *labels;
    uint32_t size;
} mnist_dataset_t;

class Dataset
{
    const char *train_images_file = "dat/train-images-idx3-ubyte";
    const char *train_labels_file = "dat/train-labels-idx1-ubyte";
    const char *test_images_file = "dat/t10k-images-idx3-ubyte";
    const char *test_labels_file = "dat/t10k-labels-idx1-ubyte";

    mnist_dataset_t *train_dataset;
    mnist_dataset_t *test_dataset;

    Dataset();
    ~Dataset();

    mnist_dataset_t *get_train_dataset();
    mnist_dataset_t *get_test_dataset();

    mnist_dataset_t *get_train_sample(int index);
    mnist_dataset_t *get_test_sample(int index);

    int get_train_size();
    int get_test_size();

private:
    mnist_dataset_t *read_dataset(const char *image_path, const char *label_path);
    void free_dataset(mnist_dataset_t *dataset);
};

#endif // DATASET_H