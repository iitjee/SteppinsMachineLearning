https://www.tensorflow.org/versions/master/tutorials/image_recognition



resos:
  http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
  http://neuralnetworksanddeeplearning.com/chap6.html
  

Researchers have demonstrated steady progress in computer vision by validating their work against ImageNet -- an academic benchmark for
computer vision. Successive models continue to show improvements, each time achieving a new state-of-the-art result: QuocNet, AlexNet,
Inception (GoogLeNet), BN-Inception-v2. 

Inception-v3 is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012. This is a standard task in computer 
vision, where models try to classify entire images into 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher". \
To compare models, we examine how often the model fails to predict the correct answer as one of their top 5 guesses -- termed "top-5 error 
rate".

AlexNet achieved by setting a top-5 error rate of 15.3% on the 2012 validation data set; Inception (GoogLeNet) achieved 6.67%; BN-
Inception-v2 achieved 4.9%; Inception-v3 reaches 3.46%.

 (http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/ 
 Andrej Karpathy who attempted to measure his own performance. He reached 5.1% top-5 error rate.)
 
 This tutorial will teach you how to use Inception-v3. You'll learn how to classify images into 1000 classes in Python or C++. We'll also 
 discuss how to extract higher level features from this model which may be reused for other vision tasks.


classify_image.py downloads the trained model from tensorflow.org when the program is run for the first time.
If you wish to supply other JPEG images, you may do so by editing the --image_file argument.
If you download the model data to a different directory, you will need to point --model_dir to the directory used.







https://www.tensorflow.org/versions/master/tutorials/image_retraining
resos: https://arxiv.org/pdf/1310.1531v1.pdf

Modern object recognition models have millions of parameters and can take weeks to fully train. Transfer learning is a technique that 
shortcuts a lot of this work by taking a fully-trained model for a set of categories like ImageNet, and retrains from the existing 
weights for new classes. 
Though it's not as good as a full training run, this is surprisingly effective for many applications, and can be run in as little as 
thirty minutes on a laptop, without requiring a GPU.


Before you start any training, you'll need a set of images to teach the network about the new classes you want to recognize.
    cd ~
    curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
    tar xzf flower_photos.tgz

from the root of your TensorFlow source directory:
bazel build tensorflow/examples/image_retraining:retrain
(if you have a machine which supports the AVX instruction set (common in x86 CPUs produced in the last few years) you can improve the 
running speed
The retrainer can then be run like this:
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos

This script loads the pre-trained Inception v3 model, removes the old top layer, and trains a new one on the flower photos you've 
downloaded. None of the flower species were in the original ImageNet classes the full network was trained on. The magic of transfer 
learning is that lower layers that have been trained to distinguish between some objects can be reused for many recognition tasks 
without any alteration.
)





