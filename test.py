
# import tensorflow as tf
# import numpy as np
#
#
# X_ph = tf.placeholder('float32', [None, 10])
#
# W = tf.Variable(tf.random_uniform([10, 20]))
#
# y = tf.matmul(X_ph, W)
#
# grad = tf.gradients(ys=y, xs=X_ph)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# X = np.random.rand(3, 10)
#
# print(sess.run(grad, feed_dict={X_ph:X}))


path = '/home/data/voc2012/SegmentationObject/2008_006554.png'
import matplotlib.pyplot as plt
