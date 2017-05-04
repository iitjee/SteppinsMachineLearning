
'''
https://www.tensorflow.org/versions/r0.10/get_started/basic_usage

To use TensorFlow you need to understand how TensorFlow:
    Represents computations as graphs.
    Executes graphs in the context of Sessions.
    Represents data as tensors.
    Maintains state with Variables.
    Uses feeds and fetches to get data into and out of arbitrary operations.
  

    
computations represented as  graphs.

Nodes in the graph are called ops (short for operations). An 'op'(node) takes zero or more Tensors, performs some computation, and 
produces Tensors.

   A Tensor is a typed multi-dimensional array. For example, you can represent a mini-batch of images as a 4-D array of floating point 
   numbers with dimensions [batch, height, width, channels]. 
    
 A TensorFlow graph is a "description" of computations.
 
 To compute anything, a graph must be launched in a Session. A Session places the graph ops(nodes) onto Devices, such as CPUs or GPUs, and 
 provides methods to execute them. These methods return tensors produced by ops as numpy ndarray objects in Python, and as 
 tensorflow::Tensor instances in C and C++   
        ndarray = n-dimension array
    


Building the graph
  To build a graph start with ops(nodes) that do not need any input (source ops), such as Constant, and pass their output to other
  ops (nodes) that do computation.
    
  The TensorFlow Python library has a default graph to which ops constructors(node-makers) add nodes. (you can also explicitly manage
  multiple graphs but not needed now)
    
    '''

'''
The assign() operation in this code is a part of the expression graph just like the add() operation, so it does not actually perform the assignment until run() executes the expression.

You typically represent the parameters of a statistical model as a set of Variables. For example, you would store the weights for a neural network as a tensor in a Variable. During training you update this tensor by running a training graph repeatedly.
'''


    import tensorflow as tf
    matrix1 = tf.constant([[3., 3.]]) #Create a Constant op(node) that produces a 1x2 matrix
    matrix2 = tf.constant([[2.], [2.]]) #Create a Constant op(node) that produces a 2x1 matrix
    product = tf.matmul(matrix1, matrix2)
    
    The default graph now has three nodes: two constant() ops(nodes) and one matmul() op(node). To actually multiply the matrices, and 
    get the result of the multiplication, you must launch the graph in a session.
    
Now our graph is constructed. Now follows "Launching"
    # Launch the default graph.
       sess = tf.Session()

    # To run the matmul op(node) we call the session 'run()' method, passing 'product'
    # which represents the output of the matmul op.  This indicates to the call
    # that we want to get the output of the matmul op back.
    
    # All inputs needed by the op are run automatically by the session.  They
    # typically are run in parallel.
    
    # The call 'run(product)' thus causes the execution of three ops in the
    # graph: the two constants and matmul.
    
    # The output of the op is returned in 'result' as a numpy `ndarray` object.
    result = sess.run(product)
    print(result)
    # ==> [[ 12.]]

    # Close the Session when we're done to release resources
    sess.close()

    #You can also enter a Session with a "with" block. The Session closes automatically at the end of the with block.
    with tf.Session() as sess:
      result = sess.run(product)
      print(result)

#all cpu and gpu stuff is managed by tf. but if u want to manage urself
    with tf.Session() as sess:
      with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1, matrix2)
        ...
        
''''
'Ínteractive Usage':
Úsually we use Session object and use its Session.run() method to execute the operations.
But for interacrtive Py envs like iPython, use 'InteractiveSession' class, and the Tensor.eval() and Operation.run() methods.
This just avoids to keep a variable holding the session.
'''
        # Enter an interactive TensorFlow Session.
        import tensorflow as tf
        sess = tf.InteractiveSession()

        x = tf.Variable([1.0, 2.0])
        a = tf.constant([3.0, 3.0])

        # Initialize 'x' using the run() method of its initializer op.
        x.initializer.run()

        # Add an op to subtract 'a' from 'x'.  Run it and print the result
        sub = tf.sub(x, a)
        print(sub.eval())
        # ==> [-2. -1.]

        # Close the Session when we're done.
        sess.close()
        
'''
Tensors:
    n-dimensional array.
     A tensor has a static type, a rank, and a shape
        
    TensorFlow programs use a tensor data structure to represent all data -- only tensors are passed between operations in the computation graph. 
'''
    
    
    
    '''
        Variables:
 Variables maintain state across executions of the graph.  
 '''
    #Following example shows a variable which acts as a simple counter
# Create a Variable, that will be initialized to the scalar value 0.
    state = tf.Variable(0, name="counter") #generally we do `counter = tf.Variable(0, name="counter")`

    # Create an Op to add one to `state`.

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    
    '''
    assign(
    ref,
    value,
    validate_shape=None,
    use_locking=None,
    name=None
    )
   -  we've used only the first two arguments
   -  Update 'ref' by assigning 'value' to it.
   -  This operation outputs "ref" after the assignment is done. This makes it easier to chain operations that need to use the reset value.
    '''

    # Variables must be initialized by running an `init` Op after having
    # launched the graph.  We first have to add the `init` Op to the graph.
    init_op = tf.initialize_all_variables() #init_op node is added to the computational graph
    

    # Launch the graph and run the ops (nodes).
    with tf.Session() as sess:
# First Run the 'init' op(node)
      sess.run(init_op) #think of this node which will be at the start of 
    
# Print the initial value of 'state'
      print(sess.run(state))
    
# Run the op that updates 'state' and print 'state'.
      for _ in range(3):
        sess.run(update)  #note that during update operation everytime, sum operation is being done
        print(sess.run(state))

    # output:

    # 0
    # 1
    # 2
    # 3
    
    
