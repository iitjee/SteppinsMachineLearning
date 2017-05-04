'''
  Variables: Creation, Initialization, Saving, and Loading

When you train a model, you use variables to hold and update parameters. Variables are in-memory buffers containing tensors. They must be 
explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyze the 
model.
We'll see two classes:
  The tf.Variable class. (It has tf.Variable() constructor)
  The tf.train.Saver class.
'''

#Creation
somevar = tf.Variable([1.0, 2.0]) #[1.0, 2.0] is also a tensor, don't you think, iit? :p
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

#we're passing a tensor(an n-dim array) to the Variable constructor to construct a variable
'''
Calling tf.Variable() adds several ops to the graph:
  - A variable op that holds the variable value.
  - An initializer op that sets the variable to its initial value. This is actually a tf.assign op.
  - The ops for the initial value, such as the zeros op for the biases variable in the example are also added to the graph.

The value returned by tf.Variable() value is an instance of the Python class tf.Variable (obvious na! :/ )
'''

#Initialization
'''
already covered but we will revisit again! :)
'Variable Initializers' must be run explicitly before other ops in your model can be run.

'''
      # Create two variables.
      weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                            name="weights")
      biases = tf.Variable(tf.zeros([200]), name="biases")
      ...
      # Add an op to initialize the variables.
      init_op = tf.global_variables_initializer()

      # Later, when launching the model
      with tf.Session() as sess:
        # Run the init operation at the very start of the session
        sess.run(init_op)
        ...
        # Use the model
        ...

#Initialization from another Variable
'''
You sometimes need to initialize a variable from the initial value of another variable.

As the op added by tf.global_variables_initializer() initializes all variables in parallel you have to be careful 
when this is needed.

To initialize a new variable from the value of another variable use the other variable's initialized_value() property.

You can use the initialized value directly as the initial value for the new variable, or you can use it as any other
tensor to compute a value for the new variable.
'''
      # Create a variable with a random value.
      weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                            name="weights")
    
      # Create another variable with the same value as 'weights'.
      w2 = tf.Variable(weights.initialized_value(), name="w2") #this is like w2 = weights
      
      # Create another variable with twice the value of 'weights'
      w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")


