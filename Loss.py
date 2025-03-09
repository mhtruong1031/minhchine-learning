import numpy as np

def mse(y_pred, y_true):
    return np.mean(np.power(y_pred-y_true, 2))

def mse_prime(y_pred, y_true):
    return 2 * (y_pred-y_true) / np.size(y_true)

# TODO: implement auto-diff