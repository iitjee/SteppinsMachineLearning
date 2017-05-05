# Let's create a Gaussian curve!
# The 1 dimensional gaussian takes two parameters, the mean value, and the standard deviation, which is commonly denoted by the name sigma.
  mean = 0.0
  sigma = 1.0

# Don't worry about trying to learn or remember this formula.  I always have to refer to textbooks or check online for the exact formula.
  z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                    (2.0 * tf.pow(sigma, 2.0)))) *
      (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
  res = z.eval()
  plt.plot(res)
# if nothing is drawn, and you are using ipython notebook, uncomment the next two lines:
#%matplotlib inline
#plt.plot(res)
      
  
  Convolution

 '''Creating a 2-D Gaussian Kernel'''
# Let's store the number of values in our Gaussian curve.
  ksize = z.get_shape().as_list()[0] #ksize = kernel size
# Let's multiply the two to get a 2d gaussian
  z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
# Execute the graph
  plt.imshow(z_2d.eval())
#for some reason an error is coming^
  
  Convolving an Image with a Gaussian
  A very common operation that we'll come across with Deep Learning is convolution. We're going to explore what this means using our new 
  gaussian kernel that we've just created. For now, just think of it as a way of filtering information. We're going to effectively 
  filter our image using this Gaussian function, as if the gaussian function is the lens through which we'll see our image data. 
  
  What it will do is at every location we tell it to filter, it will average the image values around it based on what the kernel's 
  values are. 
  
  The Gaussian's kernel is basically saying, take a lot the center, a then decesasingly less as you go farther away from the 
  center. The effect of convolving the image with this type of kernel is that the entire image will be blurred. If you would like an 
  interactive exploratin of convolution, this website is great: http://setosa.io/ev/image-kernels/
  
  
  # Let's first load an image.  We're going to need a grayscale image to begin with.  skimage has some images we can play with.  If you 
  #do not have the skimage module, you can load your own image, or get skimage by pip installing "scikit-image".
from skimage import data
import numpy as np
img = data.camera().astype(np.float32) #data.camera() returns a predefined photo
plt.imshow(img, cmap='gray')
print(img.shape)
  '''
  Notice our img shape is 2-dimensional.
  For image convolution in Tensorflow, we need our images to be 4 dimensional.
  
  Remember that when we load many iamges and combine them in a single numpy array, the resulting shape has the number of images first.
N x H x W x C
(Number of Images x Image Height x Image Width x Number of Channels)
In order to perform 2d convolution with tensorflow, we'll need the same dimensions for our image. With just 1 grayscale image, this means the shape will be:
1 x H x W x 1 (since C = 1 for grayscale)
'''
  
  
  # We could use the numpy reshape function to reshape our numpy array
img_4d = img.reshape([1, img.shape[0], img.shape[1], 1])
print(img_4d.shape)

# but since we'll be using tensorflow, we can use the tensorflow reshape function:
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
print(img_4d)
'''
output: (1, 512, 512, 1)
Tensor("Reshape_2:0", shape=(1, 512, 512, 1), dtype=float32)


Instead of getting a numpy array back, we get a tensorflow tensor. This means we can't access the shape parameter like we did with the 
numpy array. But instead, we can use get_shape(), and get_shape().as_list(): 

print(img_4d.get_shape())
print(img_4d.get_shape().as_list())

output: (1, 512, 512, 1)
[1, 512, 512, 1]

'''




'''
  We'll also have to reshape our Gaussian Kernel to be 4-dimensional as well. The dimensions for kernels are slightly different
  
   Remember that the image is:
Number of Images x Image Height x Image Width x Number of Channels
we have:
Kernel Height x Kernel Width x Number of Input Channels x Number of Output Channels
Our Kernel already has a height and width of ksize so we'll stick with that for now. (ksize defined above)
The number of input channels should match the 
number of channels on the image we want to convolve.

And for now, we just keep the same number of output channels as the input channels, 
but we'll later see how this comes into play.
'''
  # Reshape the 2d kernel to tensorflow's required 4d format: H x W x I x O
  z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
  print(z_4d.get_shape().as_list())
  
  output: [100, 100, 1, 1]

  
  
  
  
  
  
