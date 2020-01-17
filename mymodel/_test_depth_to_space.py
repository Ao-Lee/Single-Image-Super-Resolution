import tensorflow as tf
import numpy as np
x_np = [[[1, 11, 111, 1111], [2, 22, 222, 2222]],[[3, 33, 333, 3333], [4, 44, 444, 4444]]]
x_np = np.array(x_np)
x_np = np.expand_dims(x_np, axis=0)
print(x_np.shape)
print(x_np)
x = tf.constant(x_np)
y = tf.depth_to_space(x, block_size=2)
with tf.Session() as sess:
    y_np = sess.run(y)
 
print(y_np.shape)
print(y_np)




