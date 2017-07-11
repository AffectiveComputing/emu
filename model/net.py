import numpy as np
import tensorflow as tf

__author__ = ["Michał Górecki", "Paweł Kopeć"]

DEFAULT_FILTERS_COUNT = 1
DEFAULT_KERNEL_SIZE = 5
DEFAULT_POOL_SIZE = 2
DEFAULT_STRIDES = 2
DEFAULT_DEEP_SIZE = 128


def net(x, layers):
    for i, layer in enumerate(layers):
        layer_type = layer.get("type")

        if layer_type is "deep":
            break
        elif layer_type is "conv":
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

    in_size = np.product(x.get_shape().as_list()[1:])
    x = tf.reshape(x, [-1, in_size])

    for j, layer in enumerate(layers[i:]):
        out_size = layer.get("out_size") or DEFAULT_DEEP_SIZE
        x = deep(x, in_size, out_size, j)
        in_size = out_size

    return x


def deep(x, in_size, out_size, index):
    x = tf.nn.relu(x)

    W = tf.get_variable(
        "W" + str(index),
        shape=[in_size, out_size],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    b = tf.get_variable(
        "b" + str(index),
        shape=[out_size, ],
        initializer=tf.constant_initializer()
    )

    return tf.matmul(x, W) + b