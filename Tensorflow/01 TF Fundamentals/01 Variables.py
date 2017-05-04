'''
  Variables: Creation, Initialization, Saving, and Loading

When you train a model, you use variables to hold and update parameters. Variables are in-memory buffers containing tensors. They must be 
explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyze the 
model.
We'll see two classes:
  The tf.Variable class. (It has tf.Variable() constructor)
  The tf.train.Saver class.
'''
