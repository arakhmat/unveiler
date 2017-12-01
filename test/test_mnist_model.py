import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.append('../unveiler')

from keras.models import load_model
from keras.datasets import mnist

from model import Model


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
        from keras.losses import categorical_crossentropy
        from keras.utils import to_categorical
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
        
        model.compile(loss=categorical_crossentropy,
                      optimizer='adam', metrics=['accuracy'])
        
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test,  10)
        
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
    
    keras_model, frames, labels = mnist_model()
    
    model = Model(keras_model)
 
    start, offset = 0, 1
    for frame in frames[start:start+offset]:
        print('Feeforwarding through the network')
        model.predict(frame)
#       
        print('Visualizing all activations')
        model.visualize(until=10, n_cols=3)
        
        print('Deconvolving first layer')
        model.deconvolve(index=1)
    