import tensorflow as tf

from const import OUT_SIZE, WIDTH, HEIGHT


def get_graph(x, filter_sizes, kernel_size):
    current_layer = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

    conv_output_size = x.get_shape()[1] * x.get_shape()[2] * filter_sizes[-1]
    for filter_size in filter_sizes:
        conv_output_size //= 4
        current_layer = tf.layers.max_pooling2d(
            inputs=tf.layers.conv2d(
                inputs=current_layer,
                filters=filter_size,
                kernel_size=[kernel_size, kernel_size],
                padding="same",
                activation=tf.nn.relu),
            pool_size=[2, 2], strides=2)

    W = tf.get_variable("W", shape=[conv_output_size, OUT_SIZE],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[OUT_SIZE, ],
                        initializer=tf.constant_initializer())

    classes = tf.matmul(tf.reshape(current_layer, [-1, W.get_shape().as_list()[0]]), W) + b

    return tf.argmax(classes, axis=1), classes
