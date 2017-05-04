'''
#Fetches

to fetch the ouput of operations (ops i.e nodes), execute(run) the graph with a run() call on the Session object
and pass in all those tensors you want to retrieve(calculate)! :)
'''
  input1 = tf.constant([3.0])
  input2 = tf.constant([2.0])
  input3 = tf.constant([5.0])
  intermed = tf.add(input2, input3)
  mul = tf.mulitply(input1, intermed)

  with tf.Session() as sess:
  #note that we don't run any `init` op as we didn't use any Variables
    result = sess.run([mul, intermed])
    print(result)

  # output:
  # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
  
'''

#Feeds
examples so far introduced tensors into the computation graph by storing them in Constants and Variables. TensorFlow also provides a feed 
mechanism for patching a tensor directly into any operation in the graph.

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
      print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
      #see, it's simple. 
      #consider print(ses.run([output])  it's senseless as input1 and input2 don't have values
      #feed_dict is the name of the argument, here you pass 7 and 2 like 'on-the-go' to input1 and input2 placeholders
      #placeholders = otherlanguage version of variables? we'll have to see! :D

    # output:
    # [array([ 14.], dtype=float32)]
    

