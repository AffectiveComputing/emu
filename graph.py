import tensorflow as tf


def get_graph(x, filter_sizes, deep_sizes, kernel_size):
    current_layer = x

    # build convolutional and pool layers
    for filter_size in filter_sizes:
        current_layer = tf.layers.max_pooling2d(
            inputs=tf.layers.conv2d(
                inputs=current_layer,
                filters=filter_size,
                kernel_size=[kernel_size, kernel_size],
                padding="same",
                activation=tf.nn.relu),
            pool_size=[2, 2], strides=2)

    # calculate conv and poll output assuming pool size [2, 2]
    prev_input_size = x.get_shape().as_list()[1] * x.get_shape().as_list()[2] * filter_sizes[-1] // (
    4 ** len(filter_sizes))

    # normalize square input shape to linear
    current_layer = tf.reshape(current_layer, [-1, prev_input_size])

    # build feed-forward layers
    for deep_size, i in zip(deep_sizes, range(len(deep_sizes))):
        W = tf.get_variable("W" + str(i), shape=[prev_input_size, deep_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b" + str(i), shape=[deep_size, ],
                            initializer=tf.constant_initializer())

        current_layer = tf.matmul(current_layer, W) + b
        prev_input_size = deep_size
        i += 1

    return tf.argmax(current_layer, axis=1), current_layer
