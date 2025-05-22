import numpy as np
import scipy.special  # to use built-in softmax function (avoid numerical instability)


#### ACTIVATION FUNCTIONS
"""
Applies to all activation functions.

Parameters:
  Z : non activated outputs
Returns:
  (A : 2d ndarray of activated outputs, df: derivative component wise)
"""

def identity(Z):
    return Z, 1


def tanh(Z):
    A = np.empty(Z.shape)
    A = 2.0 / (1 + np.exp(-2.0 * Z)) - 1  # A = np.tanh(Z)
    df = 1 - A ** 2
    return A, df


def sintr(Z):
    A = np.empty(Z.shape)
    if Z.all() < -np.pi / 2:
        A = 0
    elif Z.all() > np.pi / 2:
        A = 1
    else:
        A = np.sin(Z)
    df = np.cos(Z)
    return A, df


def sigmoid(Z):
    A = np.empty(Z.shape)
    A = 1.0 / (1 + np.exp(-Z))
    df = A * (1 - A)
    return A, df


def relu(Z):
    A = np.empty(Z.shape)
    A = np.maximum(0, Z)
    df = (Z > 0).astype(int)
    return A, df


def softmax(Z):
    return scipy.special.softmax(Z, axis=0)  # from scipy.special


#### COST FUNCTIONS
"""
Applies to all cost functions.

Parameters:
  y_hat : predicted output from the model
  y : the right answer
Returns:
  (error : scalar value, grad: gradient of cost function component wise)
"""

def MSE_cost(y_hat, y):
    n = y_hat.shape[0]
    mse = np.square(np.subtract(y_hat, y)).mean() / 2
    grad = np.subtract(y_hat, y) / n
    return mse, grad
