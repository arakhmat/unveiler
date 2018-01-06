import numpy as np

from unveiler.layers import Layer
from unveiler.plot_utils import plot, plot3D

class Model:
    def __init__(self, keras_model):
        self.layers = []
        self.deconvolvable_layers = []
        self.conv_layers_indices = [] # Indices of Conv2D layers in deconvolvable_layers
        for keras_layer in keras_model.layers:
            layer, layer_type =  Layer.factory(keras_layer)
            self.layers.append(layer)
            if layer_type == 'conv':
                self.deconvolvable_layers.append(layer)
                self.conv_layers_indices.append(len(self.deconvolvable_layers)-1)
            elif layer_type == 'maxpool':
                self.deconvolvable_layers.append(layer)

    def predict(self, x, until=None):
        plot(x)
        if until is None:
            until = len(self.layers)
        for layer in self.layers[:until]:
            x = layer.feedforward(x)
        return x

    # Deconvolve convolutional layer at index I
    def deconvolve(self, frame=None, index=0):
        if frame is not None:
            self.predict(frame)

        if index <= -1 or index >= len(self.conv_layers_indices):
            raise ValueError('Index %d is out of bounds.\n')

        index = self.conv_layers_indices[index]

        starting_layer = self.deconvolvable_layers[index]
        consequent_layers = list(reversed(self.deconvolvable_layers[:index]))
        w = starting_layer.w.copy()
        for i in range(starting_layer.w.shape[2]):
            for j in range(starting_layer.w.shape[3]):

                w.fill(0)
                w[:, :, i, j] = np.copy(starting_layer.w[:, :, i, j])

                x = starting_layer.deconvolve(w=w)
                for layer in consequent_layers:
                    x = layer.deconvolve(x=x)

                plot3D(x)
                print('----------------------------------------------------------------------------------')

    # Visualize all the activations upto layer N
    def visualize(self, frame=None, until=1, n_cols=3):
        if frame is not None:
            self.predict(frame, until)
        for layer in self.layers[:until]:
            print(layer.name)
            plot(layer.output, n_cols)


