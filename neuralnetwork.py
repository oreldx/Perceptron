import numpy as np
import numpy.typing as npt

from perceptron import Perceptron


class NeuralNetwork:
    def __init__(
        self,
        input_size: int, 
        hidden_sizes: npt.NDArray[any],
        output_size: int,
    ) -> None:
        layers = []
        self.input_size = input_size

        for hidden_size in hidden_sizes:
            layer = [Perceptron(input_size, 'relu') for _ in range(hidden_size)]
            layers.append(np.array(layer))
            input_size = hidden_size
        
        layers.append(
            [Perceptron(input_size, 'tanh') for _ in range(output_size)]
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
            for i, perceptron in enumerate(layer):
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
        return activations
