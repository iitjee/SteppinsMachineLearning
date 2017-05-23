

'''Yo! Let's start Helloworld of TF = MNIST! xD '''

softmax (multinomial logistic) regression


The actual code for this tutorial is very short, and all the interesting stuff happens in just three lines. However, it is 
very important to understand the ideas behind it: both how TensorFlow works and the core machine learning concepts. Because of 
this, we are going to very carefully work through the code.


We know that every image in MNIST is of a handwritten digit between zero and nine. So there are only ten possible things that 
a given image can be. We want to be able to look at an image and give the probabilities for it being each digit. For example, 
our model might look at a picture of a nine and be 80% sure it's a nine, but give a 5% chance to it being an eight (because of 
the top loop) and a bit of probability to all the others because it isn't 100% sure.

This is a classic case where a softmax regression is a natural, simple model. If you want to assign probabilities to an object 
being one of several different things, softmax is the thing to do, because softmax gives us a list of values between 0 and 1 
that add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert 
  that evidence into probabilities.
  
  
  
  Training
In order to train our model, we need to define what it means for the model to be good. Well, actually, in machine learning we 
typically define what it means for a model to be bad. We call this the cost, or the loss, and it represents how far off our 
model is from our desired outcome. We try to minimize that error, and the smaller the error margin, the better our model is.

One very common, very nice function to determine the loss of a model is called "cross-entropy." Cross-entropy arises from 
thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, 
from gambling to machine learning. It's defined as:
  Hy′(y)=−∑iyi′log⁡(yi)
Where y is our predicted probability distribution, and y′ is the true distribution (the one-hot vector with the digit 
labels). 
In some rough sense, the cross-entropy is measuring how inefficient our predictions are for describing the truth.




Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd 
like to use all our data for every step of training because that would give us a better sense of what we should be doing, 
but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same 
benefit.







nielsen book (chap 3)
(improve on our vanilla implementation of backpropagation) vanilla implem = basic implem?

It's that steepness which the cross-entropy buys us, preventing us from getting stuck just when we'd expect our
neuron to learn fastest, i.e., when the neuron starts out badly wrong.

