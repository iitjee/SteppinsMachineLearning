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


  

