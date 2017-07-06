import tensorflow as tf


class Net(object):

    def build_net(x, filter_sizes, deep_sizes, kernel_size):

        conv_output = Net.__build_conv_and_poll(x, filter_sizes, kernel_size)

        # calculate conv and poll output assuming pool size [2, 2]
        x_shape = x.get_shape().as_list()
        traits_num = x_shape[1] * x_shape[2] * filter_sizes[-1]

        # every polling layer decreases size by 4
        ff_input_size = traits_num // (4 ** len(filter_sizes))

        # change matrix to vector for feed-forward x
        ff_input = tf.reshape(conv_output, [-1, ff_input_size])

        output = Net.__build_feed_forward(ff_input, ff_input_size, deep_sizes)

        return tf.argmax(output, axis=1), output

    def __build_conv_and_poll(x, filter_sizes, kernel_size):

        current_layer = x

        for filter_size in filter_sizes:

            conv_layer = tf.layers.conv2d(
                inputs=current_layer,
                filters=filter_size,
                kernel_size=[kernel_size, kernel_size],
                padding="same",
                activation=tf.nn.relu

            )

            current_layer = tf.layers.max_pooling2d(
                inputs=conv_layer,
                pool_size=[2, 2], strides=2
            )

        return current_layer

    def __build_feed_forward(x, x_size, deep_sizes):

        current_layer = x

        for deep_size, index in zip(deep_sizes, range(len(deep_sizes))):
            W = tf.get_variable(
                "W" + str(index),
                shape=[x_size, deep_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            b = tf.get_variable(
                "b" + str(index),
                shape=[deep_size, ],
                initializer=tf.constant_initializer()
            )

            current_layer = tf.matmul(current_layer, W) + b
            x_size = deep_size
            index += 1

        return current_layer
