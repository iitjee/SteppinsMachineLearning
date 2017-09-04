Keras is like front end and Tensorflow/Theano like backend
The best part of Keras is that the 'best-practices' are built-in
It also comes with pre-trained models for image recognition! How cool is that ;)


Tensorflow = Google
Theano = University of Montreal (MILA)

Tensorflow is low-level, but has more control.
Keras is high-level abstraction => easier to use

When to use Tensorflow?
1. when you try new models
2. when you build large models


When to use Keras?
1. for education / trying new things
2. for quick prototyping



Supervised Learning:
1. Choose a Model
2. Training Phase (using training data)
3. Testing Phase
4. Evaluation Phase



Keras Sequenctial Model API:
 - create an empty sequential model object and then add layers to it in sequence
            model = keras.models.Sequential()
            model.add(Dense(32, input_dim=9)) #32 is number of neurons in that layer
            model.add(Dense(128))
            
- Customizing Layers
Let's talk about the different ways we can customize a neural network layer. Before values flow from nodes in one layer to the next, they pass through an activation function.
Keras lets us choose which activation function is used for each layer by passing in the name of the activation function we want to use. In this case, I've told it to use a rectified linear unit, or RELU, activation function
        model.add(Dense(number_of_neurons, activation='reulu'))
        
        
 So far we've talked about densely connected layers which are the most basic type of layer, but Keras also supports many different types of neural network layers.
 Let's look at two other major types of layers that Keras supports. 
 
 First are convolutional layers. These are typically used to process images or spacial data. Next are recurrent layers.
                keras.layers.convolutional.conv2D()
Recurrent layers are special layers that have a memory built into each neuron. These are used to process sequential data like words in a
sentence where the previous data points are important to understanding the next data point. You can mix layers of different types in the same model as needed.
                keras.layers.recurrent.LSTM()


The final step of defining a model is to compile it by calling model.compile.
                model.compile(optimizer='adam', loss='mse')


When you compile a model, you have to pass in the optimizer algorithm and the loss function you want to use. The optimizer algorithm is the algorithm used to train your neural network.
The loss function is how the training process measures how right or how wrong your neural network's predictions are. In this case, I've used the adam optimizer function which is a common and powerful optimizer, and the mean squared error loss function.




