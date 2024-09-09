//
//  main.cpp
//  MnistTest
//
//  Created by Gonzalo Reynaga Garcia on 30/04/2023.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include "readFiles.h"
#include "NeuralNetwork.h"


std::vector<float> one_hot_encode(int label, int num_classes) {
    std::vector<float> encoded(num_classes, -0.8);
    encoded[label] = 0.8;
    return encoded;
}

void testSamples(float num_images, std::vector<std::vector<float>>& images, std::vector<int>& labels, NeuralNetwork& nn) {
    
    float total_loss = 0.0;
    int correct_predictions = 0;

    for (int i = 0; i < num_images; ++i) {
        std::vector<float> output = nn.forward(images[i]);
        std::vector<float> target = one_hot_encode(labels[i], 10);

        int predicted_label = (int) std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predicted_label == labels[i]) {
            correct_predictions++;
        }

        for (int k = 0; k < 10; ++k) {
            total_loss += 0.5 * (target[k] - output[k]) * (target[k] - output[k]);
        }
    }

    std::cout << std::endl << "Test 10k" << " - Loss: " << total_loss / num_images << " - Accuracy: " << static_cast<float>(correct_predictions) / num_images << std::endl;
    
}

int main(int argc, const char * argv[]) {
    int num_images, image_size, num_labels;
    int t10k_num_images, t10k_image_size, t10k_num_labels;

    std::vector<std::vector<float>> train_images = read_mnist_images("train-images.idx3-ubyte", num_images, image_size);
    std::vector<int> train_labels = read_mnist_labels("train-labels.idx1-ubyte", num_labels);

    std::vector<std::vector<float>> t10k_images = read_mnist_images("t10k-images.idx3-ubyte", t10k_num_images, t10k_image_size);
    std::vector<std::vector<float>> t10k_inv_images = read_mnist_images("t10k-images.idx3-ubyte", t10k_num_images, t10k_image_size, true);
    std::vector<int> t10k_labels = read_mnist_labels("t10k-labels.idx1-ubyte", t10k_num_labels);
    
    TriangleWave activation;
    
    NeuralNetwork nn( image_size, { 128, 10 }, &activation);
    
    int epochs = 20;
    float learning_rate = activation.learnRate;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        int correct_predictions = 0;

        for (int i = 0; i < num_images; ++i) {
            std::vector<float> output = nn.forward(train_images[i]);
            std::vector<float> target = one_hot_encode(train_labels[i], 10);

            nn.backward(train_images[i], target, learning_rate);

            int predicted_label = (int ) std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            if (predicted_label == train_labels[i]) {
                correct_predictions++;
            }
            
            for (int k = 0; k < 10; ++k) {
                total_loss += 0.5 * (target[k] - output[k]) * (target[k] - output[k]);
            }
        }
    
//        nn.printGradients();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / num_images << " - Accuracy: " << static_cast<float>(correct_predictions) / num_images << std::endl;
        
        if(((epoch + 1) % 10) == 0) {
            testSamples(t10k_num_images, t10k_images, t10k_labels, nn);
        }
        
    }
    
    nn.saveWeights("test.txt");
    testSamples(t10k_num_images, t10k_images, t10k_labels, nn);


    return 0;
}



