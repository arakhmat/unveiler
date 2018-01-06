import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.append('../unveiler')

import h5py
import keras.backend as K
from keras.models import load_model

from model import Model

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

if __name__ == "__main__":

    keras_model = load_model('example_model.h5',
         {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    with h5py.File('example_data.h5', 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]

    model = Model(keras_model)

    start, offset = 0, 1
    for frame in frames[start:start+offset]:
        print('Feeforwarding through the network')
        model.predict(frame)
#
        print('Visualizing all activations')
        model.visualize(until=20, n_cols=3)

        print('Deconvolving first layer')
        model.deconvolve(index=1)