//
//  readFiles.h
//  MnistTest
//
//  Created by Gonzalo Reynaga Garcia on 06/06/2024.
//

#ifndef readFiles_h
#define readFiles_h


#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

std::vector<std::vector<float>> read_mnist_images(const std::string &path, int &num_images, int &image_size, bool inverse = false) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        int32_t magic_number = 0;
        int32_t n_images = 0;
        int32_t n_rows = 0;
        int32_t n_cols = 0;

        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&n_images), 4);
        file.read(reinterpret_cast<char*>(&n_rows), 4);
        file.read(reinterpret_cast<char*>(&n_cols), 4);

        magic_number = __builtin_bswap32(magic_number);
        n_images = __builtin_bswap32(n_images);
        n_rows = __builtin_bswap32(n_rows);
        n_cols = __builtin_bswap32(n_cols);

        num_images = n_images;
        image_size = n_rows * n_cols;
//        image_size = exp2((int)(log2(n_rows * n_cols) + 1));
        std::vector<std::vector<float>> images(n_images, std::vector<float>(image_size, 0.1f));
        for (int i = 0; i < n_images; ++i) {
            for (int j = 0; j < n_rows * n_cols; ++j) {
                unsigned char temp = 0;
                file.read(reinterpret_cast<char*>(&temp), 1);
                if (inverse) {
                    images[i][j] = (1.0 - (static_cast<float>(temp) / 255.0)) * 0.8 + 0.1;
                } else {
                    images[i][j] = (static_cast<float>(temp) / 255.0) * 0.8 + 0.1;
                }
            }
        }
        return images;
    } else {
        std::cerr << "Unable to open file " << path << std::endl;
        exit(1);
    }
}

std::vector<int> read_mnist_labels(const std::string &path, int &num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        int32_t magic_number = 0;
        int32_t n_labels = 0;

        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&n_labels), 4);

        magic_number = __builtin_bswap32(magic_number);
        n_labels = __builtin_bswap32(n_labels);

        num_labels = n_labels;

        std::vector<int> labels(n_labels);
        for (int i = 0; i < n_labels; ++i) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), 1);
            labels[i] = static_cast<int>(temp);
        }
        return labels;
    } else {
        std::cerr << "Unable to open file " << path << std::endl;
        exit(1);
    }
}


#endif /* readFiles_h */
