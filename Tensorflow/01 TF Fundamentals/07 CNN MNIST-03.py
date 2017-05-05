https://www.tensorflow.org/tutorials/layers



The TensorFlow layers module provides a high-level API that makes it easy to construct a neural network.

It provides methods that facilitate the creation of dense (fully connected) layers and convolutional layers, adding activation functions, 
and applying dropout regularization.

Now, let's learn how to use layers to build a convolutional neural network model to recognize the handwritten digits in the MNIST data 
set.
The MNIST dataset comprises 60,000 training examples and 10,000 test examples of the handwritten digits 0–9, formatted as 28x28-pixel 
monochrome images


Intro to CNN:
Convolutional neural networks (CNNs) are the current state-of-the-art model architecture for image classification tasks.

CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features, which the model can then use 
for classification. 

CNNs contains three components:
    Convolutional layers, which apply a specified number of convolution filters to the image. For each subregion, the layer performs 
a set of mathematical operations to produce a single value in the output feature map. Convolutional layers then typically apply a ReLU 
activation function to the output to introduce nonlinearities into the model. === conv2d()

    Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature 
map in order to decrease processing time. A commonly used pooling algorithm is max pooling, which extracts subregions of the feature 
map (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other values. === max_pooling2d()

    Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and downsampled 
by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer. === dense()




CNN Clasifier architecture for MNIST problem:
Let's build a model to classify the images in the MNIST dataset using the following CNN architecture:
- Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
- Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
- Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
- Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
- Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped 
during training)
- Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

The tf.layers module contains methods to create each of the three layer types above:
- conv2d(). Constructs a two-dimensional convolutional layer. Takes number of filters, filter kernel size, padding, and activation function as arguments.
- max_pooling2d(). Constructs a two-dimensional pooling layer using the max-pooling algorithm. Takes pooling filter size and stride as arguments.
- dense(). Constructs a dense layer. Takes number of neurons and activation function as arguments.

Each of these methods accepts a tensor as input and returns a transformed tensor as output. This makes it easy to connect one layer to 
another: just take the output from one layer-creation method and supply it as input to another


