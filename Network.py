import numpy as np

from Layers      import *
from Activations import *
from Loss        import *

class NeuralNetwork:
    def __init__(self, layers: list) -> None:
        self.network = layers

    def train(self, x_train: np.array, y_train: np.array, epochs: int, learning_rate: float, loss_type: str = "mse", verbose: bool = True) -> None:
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)

                if loss_type == "mse":
                    loss = mse
                    loss_prime = mse_prime
                
                error    += loss(output, y)
                gradient = loss_prime(output, y)

                for layer in reversed(self.network):
                    gradient = layer.backward(gradient, learning_rate)
        
            error /= len(x_train)
            if verbose:
                print(f"Epoch {e+1}/{epochs}: error = {error}")

    def predict(self, input: np.array) -> np.array:
        output = input
        for layer in self.network:
            output = layer.forward(output)
        
        return output
    
    def __learn(self, gradient, learning_rate):
        pass