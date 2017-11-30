import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.layers
#from model import fmeasure, recall, precision

from keras.datasets import mnist

from model import Model
from metrics import fmeasure, recall, precision
'''
https://arxiv.org/pdf/1311.2901.pdf
https://www.quora.com/How-does-a-deconvolutional-neural-network-work
Decide which filter activation you want to visualize. Pass the image forward through the conv net, 
up to and including the layer where your chosen activation is.

Zero out all filter activations (channels) in the last layer except the one you want to visualize.

Now go back to image space, but through the deconv net. 
For this, the authors propose 'inverse' operations of the three common operations seen in conv nets.
Unpooling (see also bottom part of Figure 1): 
    Max pooling cannot be exactly inverted. 
    So the authors propose to remember the position of the max lower layer activation in 'switch variables'. 
    While going back, the activation from the upper layer is copy-pasted to the position pointed to by the switch variable, 
    and all other lower layer activations are set to zero. Note that different images will produce different patterns of activations, 
    so the values of the switch variables will change according to image.
ReLU: The inverse of the ReLU function is the ReLU function. 
    It sounds a bit odd, but the authors' argument is that since convolution is applied to rectified activations in the forward pass,
    deconvolution should also be applied to rectified reconstructions in the backward pass.
Deconvolution: This uses the same filters are the corresponding conv layer; 
    the only difference is that they are flipped horizontally and vertically.

Follow these three steps till you reach the image layer. 
The pattern that emerges in the image layer is the discriminative pattern that the selected activation is sensitive to. 
These are the greyish patches shown in Figure 2 in the paper.

The real-world image patches shown in Figure 2 besides the greyish patches are just crops of the input image, 
made by the receptive field of the chosen activation. 
'''






def mnist_model():
    
    model_file_name = 'mnist_model.h5'
    
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
    if os.path.exists(model_file_name):
        print('Model already exists. Loading...')
        model = load_model(model_file_name)
    else:
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        model = Sequential([
                    Conv2D(16, 5, activation='relu', name='conv1', 
                           data_format='channels_first',
                           input_shape=(1, 28, 28)),
                    MaxPooling2D(2, name='pool1', data_format='channels_first'),
                    Conv2D(32, 5, activation='relu', name='conv2', 
                           data_format='channels_first'),
                    MaxPooling2D(2, name='pool2', data_format='channels_first'),
                    Dropout(0.4, name='conv4_dropout'),
                    Flatten(name='flatten'),
                    Dense(128, activation='relu', name='dense1'),
                    Dropout(0.3, name='dense1_dropout'),
                    Dense(64, activation='relu', name='dense2'),
                    Dense(10,  activation='softmax', name='out')
                    ])
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        model.fit(x_train, y_train,
              batch_size=128,
              epochs=200,
              verbose=1,
              validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        model.save(model_file_name)
    
    return model, x_train, y_train
    

if __name__ == "__main__":
    
    test_mnist = False
    
    if test_mnist:
        keras_model, frames, labels = mnist_model()
    else:
        keras_model = load_model('test_model.h5', 
                                 {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    with h5py.File('test_data.h5', 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]
    
    model = Model(keras_model)
 
    start, offset = 0, 1
    for frame in frames[start:start+offset]:
#        frame = frames[0]
#        np_pred = model.predict(frame)
#        keras_pred = keras_model.predict(np.expand_dims(frame, axis=0))
#        assert(np.allclose(keras_pred, np_pred, atol=1e-5, rtol=1e-5))
        
        model.deconvolve(frame, index=1)
        model.visualize(until=3, n_cols=8)
    