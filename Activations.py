import numpy as np

from Layers import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation       = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate): 
        return np.multiply(output_gradient, self.activation_prime(self.input)) #dE/dY * dY/dX = dE/dX and returns error gradient with respect to input
    
class Tanh(Activation):
    def __init__(self):
        tanh       = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self) -> None:
        relu       = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)

class SoftMax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def softmax(self, input):
        exp_values = np.exp(input - np.max(input, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def softmax_prime(self, input):
        s = self.softmax(input).reshape(-1, 1)  # get column vector
        return np.diagflat(s) - np.dot(s, s.T)
    
    def backward(self, output_gradient, learning_rate):
        return np.dot(self.activation_prime(self.input), output_gradient)