
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
        


    
    
