//
//  Activation.h
//  Mnist_Multi_Layers
//
//  Created by Gonzalo Reynaga Garcia on 24/06/2024.
//

#ifndef Activation_h
#define Activation_h

class AFunction {
public:
    enum Type {
        Sigmoid,
        Gauss,
        CosWave,
        LRelu,
        Triangle,
        TriangleWave
    };
    AFunction(float defaultLearnRate, float ealpha, float ebias)
        :learnRate(defaultLearnRate), alpha(ealpha), bias(ebias)
    {
    }
public:
    virtual float eval(float z) = 0;
    virtual float derivative(float z, float y) = 0;
    virtual ~AFunction() {}
    virtual Type getType() = 0;
    float learnRate;
    float alpha;
    float bias;
};

class Sigmoid : public AFunction {
public:
    Sigmoid() : AFunction(0.02, 2.0, 0.0) { }
    
    float eval(float z) {
        return 1.0 / (1.0 + exp(-z) );
    }

    float derivative(float z, float y) {
        return y * (1.0 - y);
    }
    
    Type getType() {
        return Type::Sigmoid;
    }
};

class Gauss : public AFunction {
public:
    Gauss() : AFunction(0.01, 1.0, 0.0) {}
    
    float eval(float z) {
        return exp(- z * z );
    }

    float derivative(float z, float y) {
        return -2.0 * z * y;
    }
    
    Type getType() {
        return Type::Gauss;
    }
};


class CosWave : public AFunction {
public:
    CosWave() : AFunction(0.01, M_PI, 0.5) {}
    float eval(float z) {
        return (1.0 - cos(z)) / 2.0;
    }

    float derivative(float z, float y) {
        return sin(z) / 2.0;
    }
    
    Type getType() {
        return Type::CosWave;
    }
};

class LRelu : public AFunction {
public:
    LRelu() : AFunction(0.01, 1.0, 0.0) {}
    float eval(float z) {
        return (z > 0.0) ? z : z * 0.01;
    }

    float derivative(float z, float y) {
        return (z > 0.0) ? 1.0 : 0.01;
    }
    
    Type getType() {
        return Type::LRelu;
    }
};

class Triangle : public AFunction {
public:
    Triangle() : AFunction(0.001, 1.0, 0.0) {}
    float eval(float z) {
        return (z > 0.0) ? -z + 1.0 : z + 1.0;
    }

    float derivative(float z, float y) {
        return (z > 0.0) ? -1.0 : 1.0;
    }

    Type getType() {
        return Type::Triangle;
    }
};

class TriangleWave : public AFunction {
public:
    TriangleWave() : AFunction(0.001, 1.0, 0.0) {}
    float eval(float z) {
        double lz;
        lz = z/4.0 - floor(z/4.0) - 0.5;
        return (lz > 0.0) ? -lz + 1.0 : lz + 1.0;
    }

    float derivative(float z, float y) {
        double lz;
        lz = z/4.0 - floor(z/4.0) - 0.5;
        return (lz > 0.0) ? -1.0 : 1.0;
    }

    Type getType() {
        return Type::TriangleWave;
    }
};




#endif /* Activation_h */
