

tf.contrib.learn is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:
    running training loops
    running evaluation loops
    managing data sets
    managing feeding


tf.contrib.learn defines many common models.



#Notice how much simpler the linear regression program becomes with tf.contrib.learn:

  import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
  import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
  # other types of columns that are more complicated and useful.
  features = [tf.contrib.layers.real_valued_column("x", dimension=1)] //x is name

'''
real_valued_column(
    column_name,
    dimension=1,
    default_value=None,
    dtype=tf.float32,
    normalizer=None
)
'''

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference, testing). 
#There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. 
#The following code provides an estimator that does linear regression.
  estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
 # Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
  x = np.array([1., 2., 3., 4.])
  y = np.array([0., -1., -2., -3.])
  input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                                num_epochs=1000)
    #didn't find docs for tf.contrib.learn.io.numpy_input_fn :/

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
  estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate 'validation and testing' - data set to avoid overfitting.
  print(estimator.evaluate(input_fn=input_fn))
 
# When run, it produces      {'global_step': 1000, 'loss': 1.9650059e-11}










'''
#A Custom model
tf.contrib.learn does not lock you into its predefined models. 

Suppose we wanted to create a custom model that is not built into TensorFlow. 
We can still retain the high level abstraction of data set i.e feeding, training, etc. of tf.contrib.learn.

For illustration, we will show how to implement our own equivalent model to LinearRegressor using our knowledge of the lower 
level TensorFlow API.

To define a custom model that works with tf.contrib.learn, we need to use tf.contrib.learn.Estimator. 

tf.contrib.learn.LinearRegressor is actually a sub-class of tf.contrib.learn.Estimator. 

Instead of sub-classing Estimator, we simply provide Estimator a function model_fn that tells tf.contrib.learn how it can evaluate predictions, training steps, and loss. 
'''
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature

def model(features, labels, mode):
    # Build a linear model and predict values
    
  W = tf.get_variable("W", [1], dtype=tf.float64) #get_variable here is merely constructing a new variable. tf.Variable() 
#will also do the same job, but it's a lower-level function. for more 
# http://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow 
# https://www.tensorflow.org/api_docs/python/tf/get_variable

  b = tf.get_variable("b", [1], dtype=tf.float64) #[1] is shape of new variable
  y = W*features['x'] + b

# Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
    
# Training sub-graph
  global_step = tf.train.get_global_step()
    #get_global_step(graph=None) is the method interface
    #arg: graph in which global step is to be found. if ntn is passed, takes defaultGraph
    '''global_step refer to the number of batches seen by the graph. Everytime a batch is provided, the
    weights are updated in the direction that minimizes the loss. global_step just keeps track of the number 
    of batches seen so far. When it is passed in the minimize() argument list, the variable is increased by one. 
    Have a look at optimizer.minimize().

    You can get the global_step value using tf.train.global_step().
    The 0 is the initial value of the global step in this context
    '''
    
    
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
    
# ModelFnOps connects subgraphs we built to the
# appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))

