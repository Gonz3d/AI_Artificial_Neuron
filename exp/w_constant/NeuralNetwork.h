//
//  NeuralNetwork.h
//  MnistTest
//
//  Created by Gonzalo Reynaga Garcia on 06/06/2024.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include <vector>
#include <cmath>
#include <random>
#include "Activation.h"
#include "Layer.h"


class NeuralNetwork {
public:
    NeuralNetwork(int numOfInputs, const std::vector<int> layers, AFunction* activeFunction) {
        this->activeFunction = activeFunction;
        
        layer.push_back(new Layer(numOfInputs, layers[0], activeFunction));
        for (int i = 1; i < layers.size(); i++) {
            layer.push_back(new Layer(layers[i - 1], layers[i], activeFunction));
        }
    }

    std::vector<float> forward(const std::vector<float> &input) {
        layer[0]->eval(input);
        for (int L = 1; L < layer.size(); L++) {
            layer[L]->eval(layer[L - 1]->Y);
        }
        
        return layer.back()->Y;
    }

    void backward(const std::vector<float> &input, const std::vector<float> &target, float learningRate) {
        std::vector<float>& output =  layer[layer.size() - 1]->Y;
        std::vector<float> dOut(output.size());
        std::vector<float> *dE;
        
        // calculate Output error derivative
        for (int i = 0; i < target.size(); i++) {
            dOut[i] = 2.0 * (output[i] - target[i]);
        }
        dE = &dOut;
        
        for (int L = (int)layer.size() - 1; L > 0; L--){
            dE = layer[L]->updateWeights(layer[L - 1]->Y, learningRate, *dE);
        }
        layer[0]->updateWeights(input, learningRate, *dE);
        
    }

    ~NeuralNetwork() {
        for (Layer* ilayer : layer) {
            delete ilayer;
        }
    }
    
    void setFastMode(bool fast) {
        for (int L = 0; L < layer.size(); L++) {
            layer[L]->fast = fast;
        }
    }
    
    void printGradients() {
        for (int L = 0; L < layer.size(); L++) {
            std::cout <<"Layer " << L << std::endl;
            for(int g = 0; g < layer[L]->dE_dX.size(); g++) {
                std::cout << std::fixed << std::setw(11) << std::setprecision(6) << layer[L]->dE_dX[g] <<", ";
            }
            std::cout << std::endl;
        }
    }
    
    void saveWeights(std::string filename) {
        std::ofstream file;
        file.open (filename);
        for (int L = 0; L < layer.size(); L++) {
            file << "Layer " << L << std::endl;
            for(int n = 0; n < layer[L]->node.size(); n++) {
                file << layer[L]->node[n].beta;
                for(int w = 0; w < layer[L]->node[n].W.size(); w++) {
                    file << "," << layer[L]->node[n].W[w];
                }
                file << std::endl;
            }
        }
        file.close();
    }



private:
    std::vector<Layer*> layer;
    AFunction* activeFunction;
};



#endif /* NeuralNetwork_h */
