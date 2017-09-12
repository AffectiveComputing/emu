import numpy as np
import tensorflow as tf


DEFAULT_FILTERS_COUNT = 1
DEFAULT_KERNEL_SIZE = 5
DEFAULT_POOL_SIZE = 2
DEFAULT_STRIDES = 2
DEFAULT_FC_SIZE = 128


def net(in_data, layers, fc_dropout, conv_dropout):
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
    x = in_data

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
            print(x.shape)
            x = tf.nn.dropout(
                x, 1 - conv_dropout,
                noise_shape=[tf.shape(x)[0], 1, 1, tf.shape(x)[-1]]
            )
            print(x.shape)
        else:
            break

    x_shape = x.get_shape().as_list()

    # unroll input to fit deep layer
    if 3 < len(x_shape):
        # multiply all dimensions except for the first that represents the
        # number of input vectors
        in_size = np.product(x_shape[1:])
        x = tf.reshape(x, [-1, in_size])

    for j, layer in enumerate(layers[i:]):
        x = build_fc_layer("fc_{}".format(j), x, in_size, layer["out_size"],
                           fc_dropout, j == len(layers[i:]) - 1)
        in_size = layer.get("out_size", DEFAULT_FC_SIZE)

    return x


def build_fc_layer(name, in_data, in_size, out_size, dropout, is_out):
    """ Build a fc layer on the top of previous output. """
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[out_size, ],
                            initializer=tf.constant_initializer())
    if not is_out:
        return tf.nn.dropout(tf.nn.relu(tf.matmul(in_data, W) + b), 1 - dropout)
    else:
        return tf.matmul(in_data, W) + b
