import numpy as np
import numpy.typing as npt
from pprint import pprint

from perceptron import Perceptron


class NeuralNetwork:
    def __init__(
        self,
        input_size: int, 
        hidden_layers: list,
        output_layer: tuple,
    ) -> None:
        layers = []
        self.input_size = input_size

        for hidden_size, hidden_activation in hidden_layers:
            layer = [Perceptron(input_size, hidden_activation) for _ in range(hidden_size)]
            layers.append(np.array(layer))
            input_size = hidden_size
        
        output_size = output_layer[0]
        output_activation = output_layer[1]
        layers.append(
            [Perceptron(input_size, output_activation) for _ in range(output_size)]
        )
        self.layers = np.array(layers, dtype=object)

    def fit(
        self,
        inputs: npt.NDArray[np.float64], 
        target: float,
        lr: float
    ):

        activations = np.asarray(inputs)

        for idx, layer in enumerate(self.layers):
            layer_activiations = []
            # If last layer
            if (idx+1 == len(self.layers)):
                if target == 0.:
                    targets = [1., 0., 0.]
                elif target == 0.5:
                    targets = [0., 1., 0.]
                elif target == 1.:
                    targets = [0., 0., 1.]
                for perceptron, target in zip(layer, targets):
                    prediction = perceptron.predict(activations)
                    perceptron.fit(activations, target, lr)
            else:
                for perceptron in layer:
                    prediction = perceptron.predict(activations)
                    perceptron.fit(activations, target, lr)
                    layer_activiations.append(prediction)
            activations = np.asarray(layer_activiations)


    def predict(
        self,
        inputs: npt.NDArray[np.float64], 
    ):
        activations = np.asarray(inputs)
        for layer in self.layers:
            activations = [perceptron.predict(activations) for perceptron in layer]
        return np.array(activations)
