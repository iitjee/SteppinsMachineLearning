#Kaden

''' numpy version '''
  x = np.linspace(-3.0, 3.0, 100)
# Immediately, the result is given to us.  An array of 100 numbers equally spaced from -3.0 to 3.0.
  print(x)
# We know from numpy arrays that they have a `shape`, in this case a 1-dimensional array of 100 values
  print(x.shape)
# and a `dtype`, in this case float64, or 64 bit floating point values.
  print(x.dtype)


'''tf version'''
  x = tf.linspace(-3.0, 3.0, 100)
  print(x)
  
  Think of tf.Tensors the same way as you would the numpy.array. It is described by its shape, in this case, only 1 dimension of 100 
  values. And it has a dtype, in this case, float32. But unlike the numpy.array, there are no values printed here! That's because it 
  actually hasn't computed its values yet. Instead, it just refers to the output of a tf.Operation which has been already been added to 
  Tensorflow's default computational graph. The result of that operation is the tensor that we are returned.
  
  
#Tensor Shapes
# We can find out the shape of a tensor like so:
print(x.get_shape())

# %% Or in a more friendly format
print(x.get_shape().as_list())


#  Let's try and inspect the underlying graph.  We can request the "default" graph where all of our operations have been added:
  g = tf.get_default_graph()
  [op.name for op in g.get_operations()]
'''
So Tensorflow has named each of our operations to generally reflect what they are doing. There are a few parameters that are all 
prefixed by LinSpace, and then the last one which is the operation which takes all of the parameters and creates an output for the 
linspace.
'''  
  g.get_tensor_by_name('LinSpace' + ':0')
  #<tf.Tensor 'LinSpace:0' shape=(100,) dtype=float32>
  
  
  
#Sessions
  # We're first going to create a session:
sess = tf.Session()

  # Now we tell our session to compute anything we've created in the tensorflow graph.
computed_x = sess.run(x)
print(computed_x)

  # Alternatively, we could tell the previous Tensor to evaluate itself using this session:
computed_x = x.eval(session=sess)
print(computed_x)

  # We can close the session after we're done like so:
sess.close()

  #We could also explicitly tell the session which graph we want to manage:
sess = tf.Session(graph=g)
sess.close()
  #By default, it grabs the default graph. But we could have created a new graph like so:  
  g2 = tf.Graph()
  
  
#Interactive Session (already discussed in prev)
sess = tf.InteractiveSession()
x.eval()
  
