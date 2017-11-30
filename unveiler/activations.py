import numpy as np


class Activation():
    idx = 0
    def __init__(self, name):
        self.name = name
        
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def softmax(x):
        x = np.exp(x)
        return x / x.sum()
    
    def __call__(self, x):
        if self.name == 'relu':
            return Activation.relu(x)
        elif self.name == 'sigmoid':
            return Activation.sigmoid(x)
        elif self.name == 'softmax':
            return Activation.softmax(x)
        else:
            return x
    
    