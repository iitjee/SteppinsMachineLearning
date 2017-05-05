# Let's create a Gaussian curve!
# The 1 dimensional gaussian takes two parameters, the mean value, and the standard deviation, which is commonly denoted by the name sigma.
  mean = 0.0
  sigma = 1.0

# Don't worry about trying to learn or remember this formula.  I always have to refer to textbooks or check online for the exact formula.
  z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                    (2.0 * tf.pow(sigma, 2.0)))) *
      (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
      
# Let's store the number of values in our Gaussian curve.
  ksize = z.get_shape().as_list()[0]
# Let's multiply the two to get a 2d gaussian
  z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
# Execute the graph
  plt.imshow(z_2d.eval())
