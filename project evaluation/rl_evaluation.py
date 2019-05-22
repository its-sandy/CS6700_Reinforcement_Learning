#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

matA = tf.placeholder(tf.float32, shape = (None, None))
def customOps(n):
    """
    Define custom tensorflow operations
    """
    # Define placeholder
    # perform tf operations listed in list
    # Even if condition, matrix manipulaitons should be in Tensorflow

    #task 1
    mat_1_1 = tf.reverse(matA, [1])
    mat_1_1 = tf.matrix_band_part(mat_1_1, -1, 0)
    mat_1_1 = tf.reverse(mat_1_1, [1])

    mat_1_2 = tf.subtract(matA, mat_1_1)
    mat_1_1 = tf.transpose(mat_1_1)
    mat_1 = tf.add(mat_1_2, mat_1_1)

    #task 2
    m = tf.reduce_max(mat_1, axis = 1)

    #task 3
    mat_3_1 = tf.tile(tf.reshape(m, [1,n]), [n,1])
    mat_3_1 = tf.exp(mat_3_1)
    mat_3_1 = tf.reverse(mat_3_1, [0])
    mat_3_1 = tf.matrix_band_part(mat_3_1, -1, 0)
    mat_3_1 = tf.reverse(mat_3_1, [0])

    mat_3_2 = tf.reduce_sum(mat_3_1, axis = 1)
    mat_3 = mat_3_1 / tf.reshape(mat_3_2, (-1, 1))

    #task 4
    v1 = tf.reduce_max(mat_3, axis = 0)
    #task 5
    v2 = tf.reduce_max(mat_3, axis = 1)

    #task 6
    v = tf.math.softmax(tf.concat([v1, v2], axis = 0))

    #task 7
    ind = tf.argmax(v)

    #task 8
    finalVal = tf.cond(n/3 < tf.to_float(ind), lambda : tf.norm(v1 - v2), lambda : tf.norm(v1 + v2))

    return finalVal


if __name__ == '__main__':
    mat = np.asarray([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    n = mat.shape[0]

    finalVal = customOps(n)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={matA: mat})
    print(outVal)
    sess.close()