import numpy as np

class Layer:
    def __init__(self) -> None:
        self.input  = None
        self.output = None
        pass

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, n_input, n_neurons) -> None:
        self.weights = np.random.randn(n_neurons, n_input)
        self.biases  = np.random.randn(n_neurons, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        input_gradient  = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weight_gradient
        self.biases  -= learning_rate * output_gradient

        return input_gradient # input grad will be the output grad of the prior layer