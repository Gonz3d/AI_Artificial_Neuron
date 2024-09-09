//
//  Layer.h
//  Mnist_Multi_Layers
//
//  Created by Gonzalo Reynaga Garcia on 24/06/2024.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <vector>
#include <random>
#include "Activation.h"


struct Node {
    std::vector<float> W;
    float theta;
    float eval(const std::vector<float>& input) {
        float z = theta;
        for (int i = 0; i < input.size(); ++i) {
            z += input[i] * W[i];
        }
        return z;
    }
};



class Layer {
public:
    Layer(int numOfInputs, int numOfOutputs, AFunction *activeFunction) {
        this->activeFunction = activeFunction;
        this->fast = true;
        Nx = numOfInputs;
        Ny = numOfOutputs;
        node.resize(numOfOutputs, Node{
           .W = std::vector<float>(numOfInputs),
           .theta = 0
        });
        Z.resize(numOfOutputs);
        Y.resize(numOfOutputs);
        dE_dX.resize(numOfInputs);
        
        /* *************************************************************** */
        /* Init values */
        float initAlpha = activeFunction->alpha * sqrt(Nx); // Starting Alpha
        float initTheta = activeFunction->bias * initAlpha; // Starting Theta
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);
 
        for (int n = 0; n < node.size(); ++n) {
            Node& inode = node[n];
            for (int i = 0; i < numOfInputs; ++i) {
                inode.W[i] = initAlpha * d(gen) / Nx;
            }
            inode.theta = initTheta;
        }
    }

    void eval(const std::vector<float>& input) {
        for (int n = 0; n < node.size(); ++n) {
            Z[n] = node[n].eval(input);
            Y[n] = activeFunction->eval(Z[n]);
        }
    }

    std::vector<float>* updateWeights(const std::vector<float> &input, float learningRate, const std::vector<float>& dE) {
        /* *********************************************************** */
        // calculate Transfer Gradients
        std::vector<float> dE_dZ(Y.size());
        for (int n = 0; n < Y.size(); n++) {
            dE_dZ[n] = dE[n] * activeFunction->derivative(Z[n], Y[n]);
        }

        /* *********************************************************** */
        // calculate Transfer Gradients for previous layer
        // if it's the input layer, there is no need to transfer gradients

        for (int i = 0; i < input.size(); i++) {
            dE_dX[i] = 0.0;
            for (int n = 0; n < Y.size(); n++) {
                dE_dX[i] += node[n].W[i] * dE_dZ[n];
            }
        }


        /* *********************************************************** */
        // updating Weights
        for (int n = 0; n < node.size(); ++n) {
            Node& inode = node[n];
            for (int i = 0; i < input.size(); ++i) {
                inode.W[i] -= (learningRate / (fast ? 1.0f : (Nx / 2.0f))) * input[i] * dE_dZ[n] ;
            }
            inode.theta -= (learningRate) * dE_dZ[n];
        }

        return &dE_dX;
    }
    
    
public:
    std::vector<Node> node;
    std::vector<float> Z;
    std::vector<float> Y;
    std::vector<float> dE_dX;
    AFunction* activeFunction;
    float Nx;
    float Ny;
    bool fast;
};

#endif /* Layer_hpp */
