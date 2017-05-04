# some even basic fundas


'''
The central unit of data in TensorFlow is the tensor
 A tensor's rank is its number of dimensions
 '''
 
 3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3] i.e there are 2 matrices each with 1 row and 3 cols


You might think of TensorFlow Core programs as consisting of two discrete sections:
- Building the computational graph.
- Running the computational graph.


 Each node takes zero or more tensors as inputs and produces a tensor as an output. One type of node is a constant. Like all 
 TensorFlow constants, it takes no inputs, and it outputs a value it stores internally. 
 
  node1 = tf.constant(3.0, tf.float32)
  node2 = tf.constant(4.0) # also tf.float32 implicitly
  print(node1, node2)
 
 #Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
 
 Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. Instead, they are nodes that, 
 when evaluated, would produce 3.0 and 4.0, respectively. To actually evaluate the nodes, we must run the computational 
 graph within a session. A session encapsulates the control and state of the TensorFlow runtime.


The following code creates a Session object and then invokes its run method to run enough of the computational graph to evaluate node1 and node2. By running the computational graph in a session as follows:

sess = tf.Session()
print(sess.run([node1, node2]))
#[3.0, 4.0]


A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value 
  later.
  a = tf.placeholder(tf.float32)
  b = tf.placeholder(tf.float32)
  adder_node = a + b  # + provides a shortcut for tf.add(a, b)
  
The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b) and then 
an operation on them. We can evaluate this graph with multiple inputs by using the feed_dict parameter to specify Tensors 
that provide concrete values to these placeholders:
    print(sess.run(adder_node, {a: 3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
    
    #more explicitly  
    print(sess.run(adder_node, feed_dict = {a: [1,3], b: [2, 4]}))
    
    
#7.5
#[ 3.  7.]




#Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:

    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    
'''
V. V. IMPORTANT Note:
Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not 
initialized when you call tf.Variable
'''

#to initialize Variables
    init = tf.global_variables_initializer()
    ...
    ...
    sess.run(init)

#It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we 
#call sess.run, the variables are uninitialized.


  
 
 
# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x:[1,2,3,4]}))
#[ 0.          0.30000001  0.60000002  0.90000004]

'''
We've created a model, but we don't know how good it is yet. To evaluate the model on training data, we need a y 
placeholder to provide the desired values, and we need to write a loss function.

A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model for 
linear regression, which sums the squares of the deltas between the current model and the provided data. 

linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square to square that error. 

Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:
'''
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# producing the loss value 23.66




#Say, god has told you, the above model is perfect if the values of W and b are perfect values of -1 and 1.
#A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign.
  fixW = tf.assign(W, [-1.]) #think of fixW as an operation. that's tf.assign returns an operation
  fixb = tf.assign(b, [1.]) #another operation is creatd
  sess.run([fixW, fixb]) #now the operations should be run. till now the values are not actually assigned!
  print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
  
  #The final print shows the loss now is zero.
  
  
  
  #tf.train API
  '''TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest 
  optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect 
  to that variable.
  
  In general, computing symbolic derivatives (simply derivatives, wtf is symbolic) manually is tedious and error-prone. 
  (obviously!)
  
  Consequently, TensorFlow can automatically produce derivatives given only a description of the model using the function 
  tf.gradients. nice!
  
  For simplicity, optimizers typically do this for you.
  '''
  
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss) #loss is defined above. oopar dekh
  
  sess.run(init) # reset values to incorrect defaults.
  for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

  print(sess.run([W, b]))
  
  '''results in the final model parameters:
 [array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]
 '''
  '''
 Hurray!!Now we have done actual machine learning! Although doing this simple linear regression doesn't require much 
 TensorFlow core code, more complicated models and methods to feed data into your model necessitate more code. Thus 
 TensorFlow provides 
 higher level abstractions for common patterns, structures, and functionality. We will learn how to use some of these 
 abstractions in the next section. 
  '''
  
  
  
  
  #one last time = whole code
  import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#When run, it produces
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
  
  
