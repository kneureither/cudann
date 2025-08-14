#ifndef DATALOADER_H
#define DATALOADER_H

#include <fstream>
#include <iostream>
#include <bit>


#include "tensor.h"
#include "utils.h"
#include "dataset.h"




class DataLoader {
    public:
        DataLoader(const std::string& data_path, const std::string& label_path, unsigned int batch_size) {
            this->data_path = data_path;
            this->label_path = label_path;
            this->batch_size = batch_size;

            this->_load_images_from_file();
            this->_load_labels_from_file();
            this->shuffle_indices();
        }

        ~DataLoader() {
            this->_free_dataset();
        };

        void shuffle_indices();

        Tensor<float> load_data();
        Tensor<int> load_labels();
        Tensor<float> load_data_batch(int batch_idx);
        Tensor<int> load_labels_batch(int batch_idx);

        mnist_image_t *get_sample(int idx);
        uint8_t get_label(int idx);
        size_t get_size();
        size_t get_max_num_batches();
        bool shuffle = false;
        int shuffle_seed = 42;
        unsigned int batch_size;

        std::string get_image_as_string(int idx);


    private:
        std::string data_path;
        std::string label_path;
        std::vector<int> indices;

        mnist_dataset_t dataset;

        void _load_images_from_file();
        void _load_labels_from_file();
        void _free_dataset();

};

#endif