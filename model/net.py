import numpy as np
import tensorflow as tf

__author__ = ["Michał Górecki", "Paweł Kopeć"]

DEFAULT_FILTERS_COUNT = 1
DEFAULT_KERNEL_SIZE = 5
DEFAULT_POOL_SIZE = 2
DEFAULT_STRIDES = 2
DEFAULT_DEEP_SIZE = 128


def net(x, layers):
    """
    Build neural net of architecture described in layers list. Layers types
    include convolutional, pool and feed-forward (deep) layers, eg.:

    layers = [
        {"type": "conv", "filters_count": 32, "kernel_size": 5},
        {"type": "pool", "strides": 4},
        {"type": "conv", "filters_count": 128, "kernel_size": 5},
        {"type": "deep", "out_size": 100},
        {"type": "deep", "out_size": 7}
    ]

    :param x:       input placeholder
    :param layers:  list of dicts describing layers
    :return:        output variable of a net
    """

    # build convolutional and pool layers
    for i, layer in enumerate(layers):
        layer_type = layer.get("type")

        if layer_type is "conv":
            filters_count = layer.get("filters_count") or DEFAULT_FILTERS_COUNT
            kernel_size = layer.get("kernel_size") or DEFAULT_KERNEL_SIZE

            x = tf.layers.conv2d(
                inputs=x,
                filters=filters_count,
                kernel_size=[kernel_size, kernel_size],
                padding="same",
                activation=tf.nn.relu
            )
        elif layer_type is "pool":
            poll_size = layer.get("pool_size") or DEFAULT_POOL_SIZE
            strides = layer.get("strides") or DEFAULT_STRIDES

            x = tf.layers.max_pooling2d(
                inputs=x,
                pool_size=poll_size,
                strides=strides
            )
        else:
            break

    x_shape = x.get_shape().as_list()

    # unroll input to fit deep layer
    if 3 < len(x_shape):
        # multiply all dimensions except for the first that represents the
        # number of input vectors
        in_size = np.product(x_shape[1:])
        x = tf.reshape(x, [-1, in_size])

    # build deep layers
    for j, layer in enumerate(layers[i:]):
        out_size = layer.get("out_size") or DEFAULT_DEEP_SIZE
        x = deep_layers(x, in_size, out_size, j)
        in_size = out_size

    return x


def deep_layers(x, in_size, out_size, index):
    """

    :param x:           input placeholder
    :param in_size:     size of output of previous layer
    :param out_size:    size of output of current layer
    :param index:       number needed for naming tf variables
    :return:            output variable of a net
    """
    x = tf.nn.relu(x)

    # weights
    W = tf.get_variable(
        "W" + str(index),
        shape=[in_size, out_size],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    # bias
    b = tf.get_variable(
        "b" + str(index),
        shape=[out_size, ],
        initializer=tf.constant_initializer()
    )

    return tf.matmul(x, W) + b
