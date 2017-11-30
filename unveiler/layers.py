import numpy as np
import keras.layers

from activations import Activation

class Layer:    
    @staticmethod
    def factory(keras_layer):
        if isinstance(keras_layer, keras.layers.Dense):
            return Dense(keras_layer), 'other'
        elif isinstance(keras_layer, keras.layers.Conv2D):
            return Conv2D(keras_layer), 'conv'
        elif isinstance(keras_layer, keras.layers.MaxPooling2D):
            return MaxPooling2D(keras_layer), 'maxpool'
        elif isinstance(keras_layer, keras.layers.Flatten):
            return Flatten(keras_layer), 'other'
        elif isinstance(keras_layer, keras.layers.BatchNormalization):
            return BatchNormalization(keras_layer), 'other'
        elif isinstance(keras_layer, keras.layers.Dropout):
            return Dropout(keras_layer), 'other'
        else:
            raise ValueError('Layer %s is not supported' % str(type(keras_layer)))

class Dense(Layer):
    def __init__(self, keras_layer):
        
        self.w = keras_layer.get_weights()[0]
        self.b = keras_layer.get_weights()[1]
        self.activation = Activation(keras_layer.activation.__name__)
        
    def feedforward(self, x):
        return self.activation(x.dot(self.w) + self.b)

class Conv2D(Layer):
    def __init__(self, keras_layer):
        
        self.w = keras_layer.get_weights()[0]
        self.b = keras_layer.get_weights()[1]
        self.activation = Activation(keras_layer.activation.__name__)
        self.kernel_size = keras_layer.kernel_size
        self.strides = keras_layer.strides
        
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        
        # Output of deconvolution is the same size as input 
        self.deconvolutional_output = np.zeros(keras_layer.input_shape[1:], dtype=np.float32)
        
    def feedforward(self, x):
        self.output.fill(0)
        for i in range(self.output.shape[0]):
            for j in range(x.shape[0]):
                for k in range(0, self.output.shape[1], self.strides[0]):
                    for l in range(0, self.output.shape[2], self.strides[1]):
                        self.output[i, k, l] += \
                            (x[j, k:k+self.kernel_size[0], l:l+self.kernel_size[1]] \
                             * self.w[:, :, j, i]).sum()
            self.output[i, :, :] += self.b[i]
        self.output = self.activation(self.output)
        return self.output
    
    def deconvolve(self, x=None, w=None):
        
        # If this layer is the starting point of deconvolution, set x to be layer's output
        if x is None: x = self.output
        
        # Use layer's weights if it's not the starting point
        if w is None: w = self.w
        
        x = self.activation(x)
        for i in range(x.shape[0]):
            for j in range(self.deconvolutional_output.shape[0]):
                for k in range(x.shape[1]):
                    for l in range(x.shape[2]):
                        self.deconvolutional_output[j, k:k+self.kernel_size[0], l:l+self.kernel_size[1]] += \
                            (x[i, k, l] * w[:, :, j, i].transpose())
        return self.deconvolutional_output

class MaxPooling2D(Layer):
    def __init__(self, keras_layer):
        self.pool_size = keras_layer.pool_size
        self.strides = keras_layer.strides
             
        output_shape = keras_layer.output_shape[1:]
        self.output = np.zeros((output_shape), dtype=np.float32)
        
        # Keep track of indices for deconvolution
        self.indices = np.zeros(list(output_shape)+[2], dtype=np.int32)
        
        # Output of deconvolution is the same size as input 
        self.deconvolutional_output = np.zeros(keras_layer.input_shape[1:], dtype=np.float32)
       
    def feedforward(self, x):
        for idx_in in range(x.shape[0]):
            for idx_ih in range(0, x.shape[1]//self.pool_size[0]*self.pool_size[0], self.strides[0]):
                for idx_iw in range(0, x.shape[2]//self.pool_size[1]*self.pool_size[1], self.strides[1]):
                    slice_of_input = x[idx_in, idx_ih:idx_ih+self.pool_size[0], idx_iw:idx_iw+self.pool_size[1]]
                    max_value = slice_of_input.max()
                    max_index = np.where(slice_of_input == max_value)
                    idx_oh = idx_ih // self.pool_size[0]
                    idx_ow = idx_iw // self.pool_size[1]
                    self.output[idx_in, idx_oh, idx_ow] = max_value
                    self.indices[idx_in, idx_oh, idx_ow] = \
                        np.array([max_index[0][0], max_index[1][0]]) + np.array([idx_ih, idx_iw])
        return self.output
    
    def deconvolve(self, x):        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    jj, kk = self.indices[i, j, k]
                    self.deconvolutional_output[i, jj, kk] = x[i, j, k]
        return self.deconvolutional_output

class Flatten(Layer):
    def __init__(self, keras_layer):
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        
    def feedforward(self, x):
        self.output = x.flatten().reshape((1, -1))
        return self.output
    
class BatchNormalization(Layer):
    def __init__(self, keras_layer):        
        self.g = keras_layer.get_weights()[0] # gamma
        self.b = keras_layer.get_weights()[1] # beta
        self.m = keras_layer.get_weights()[2] # mean
        self.v = keras_layer.get_weights()[3] # variance
        self.e = keras_layer.epsilon
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        
    def feedforward(self, x):
        self.output = x
        for i in range(x.shape[0]):
           self.output[i] =  self.g[i] * (x[i] - self.m[i]) / np.sqrt(self.v[i] + self.e) + self.b[i]
        return self.output
    
class Dropout(Layer):
    def __init__(self, keras_layer):   
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        
    def feedforward(self, x):
        self.output = x
        return self.output