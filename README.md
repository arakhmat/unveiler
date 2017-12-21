# Unveiler
Utility for visualizing Convolutional Neural Networks

### Prerequisites
* [Numpy](https://github.com/numpy/numpy)  <br />
* [Keras](https://github.com/fchollet/keras) <br />
### Installing
Clone the repository and install it as a python module:
```
git clone https://github.com/arakhmat/unveiler
cd unveiler
pip install -e .
```
While in the same directory, test the installation by running:
```
cd test
python test_example_model.py
python test_minst_model.py
```
## How does it work
https://arxiv.org/pdf/1311.2901.pdf  
https://www.quora.com/How-does-a-deconvolutional-neural-network-work

Decide which filter activation you want to visualize. Pass the image forward through the conv net, 
up to and including the layer where your chosen activation is.

Zero out all filter activations (channels) in the last layer except the one you want to visualize.

Now go back to image space, but through the deconv net. 
For this, the authors propose 'inverse' operations of the three common operations seen in conv nets.

##### Unpooling:   
Max pooling cannot be exactly inverted. So the authors propose to remember the position of the max lower layer activation in 'switch variables'. While going back, the activation from the upper layer is copy-pasted to the position pointed to by the switch variable, and all other lower layer activations are set to zero. Note that different images will produce different patterns of activations, so the values of the switch variables will change according to image.
    
##### ReLU:   
The inverse of the ReLU function is the ReLU function. It sounds a bit odd, but the authors' argument is that since convolution is applied to rectified activations in the  forward pass, deconvolution should also be applied to rectified reconstructions in the backward pass.
    
##### Deconvolution:  
This uses the same filters are the corresponding conv layer. the only difference is that they are flipped horizontally and vertically.

Follow these three steps till you reach the image layer. The pattern that emerges in the image layer is the discriminative pattern that the selected activation is sensitive to.

## Supported Layers
* Conv2D
* MaxPool2D
* Flatten
* Dense
* BatchNormalization (for Conv2D layer only)

## Supported Activations
* Relu
* Sigmoid
* Softmax

## Known Limitations
* Only 2D convolutional neural networks are supported
* There is no BatchNormalization layer for dense networks
* Conv2D and MaxPool2D layers must have 'channel_first' data format
* Conv2D layers cannot be padded

