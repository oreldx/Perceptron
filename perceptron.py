import numpy as np
import numpy.typing as npt
from random import uniform

class Perceptron:

    def __init__(
        self,
        n: int, 
        activation_function: str,
    ) -> None:

        self.wheights = [uniform(0., 1.) for _ in range(n)]
        self.biais = uniform(-1., 1.)
        self.activation_function = activation_function

    def fit(
        self, 
        inputs: npt.NDArray[np.float64], 
        target: float,
        lr: float,
    ) -> None:

        prediction = self.predict(inputs)
        error = target - prediction
        self.wheights += lr * error * inputs
        self.biais += error * lr
        # print(f"Error: {error}")

    def predict(
        self, 
        inputs: npt.NDArray[np.float64],
    ) -> float:

        activation = np.dot(self.wheights, inputs) + self.biais
        return self.match_activation_function(activation)

    def match_activation_function(
        self, 
        x: npt.NDArray[np.float64],
    ) -> None:

        if self.activation_function == 'heaviside':
            return heaviside(x)
        elif self.activation_function == 'sigmoid':
            return sigmoid(x)
        elif self.activation_function == 'tanh':
            return tanh(x)
        elif self.activation_function == 'relu':
            return relu(x)
        else:
            print('Incorrect mentionned activation function')


# --- some utility fonctions

def heaviside(x):
    """
    Return 1 if positive otherwise return -1

    Args: 
        x (float or array)
    Returns:
        Element-wise Heaviside function activations of the input
    """
    return np.heaviside(x, 0)

def sigmoid(x):
    """
    Applies the logistic function element-wise

    Args:
        x (float or array)
    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1. / (1. + np.exp(-x))

def tanh(x):
    """
    Applies the hyperbsigmoidolic tan function element-wise

    Args:
        x (float or array)
    Returns:
        Element-wise tanh activations of the input
    """
    # return np.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    """
    Return 0 is input is negative otherwise return the input

    Args:
        x (float or array)
    Returns:
        Element-wise relu activations of the input
    """
    return np.maximum(0., x)
