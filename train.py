import tensorflow as tf

from const import *


class Trainer(object):

    def __init__(self):
        self.logs = False

    def train_graph(self, data_set, learning_rate, desired_loss, max_epochs,
                    decay_interval, decay_rate):

        x = tf.placeholder(tf.float32, IN_SHAPE)

        #TODO make it not hard coded
        prediction, output = get_graph(x, [32, 64], [OUT_SIZE], 5)

        correct = tf.placeholder(tf.int64, [BATCH_DATA_SIZE])
        loss = tf.reduce_mean(tf.abs(output - tf.one_hot(correct, OUT_SIZE)))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(prediction, correct), dtype=tf.float32))

        rate = tf.placeholder(tf.float32, [])
        train = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

        init_glob = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_glob)

            epochs = 0
            batches = 0
            to_decay = 1
            current_loss = float('Inf')

            while 0 < data_set.size:
                data, labels = data_set.get_data(BATCH_DATA_SIZE, IMG_SHAPE)
                batches += 1

                while desired_loss < current_loss:
                    epochs += 1
                    to_decay += 1

                    _, current_loss, a, p, c, o = sess.run(
                        [train, loss, accuracy, prediction, correct, output],
                        feed_dict={
                            x: data,
                            rate: learning_rate,
                            correct: labels
                        })

                    if self.logs:
                        print("Iteration ", epochs)
                        print("\tBatch ", batches)
                        print("\tLoss ", current_loss)
                        print("\tAccuracy", a)
                        print("\tLearning rate ", learning_rate)

                    if to_decay == decay_interval:
                        learning_rate *= decay_rate
                        decay_rate = 0

                    if max_epochs == epochs:
                        break

            sess.close()

    def set_logs(self, logs):
        self.logs = logs

#TODO maybe as a class
def get_graph(x, filter_sizes, deep_sizes, kernel_size):
    current_layer = x

    # build convolutional and pool layers
    for filter_size in filter_sizes:

        inputs = tf.layers.conv2d(
            inputs=current_layer,
            filters=filter_size,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu

        )

        current_layer = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=[2, 2], strides=2
        )

    # calculate conv and poll output assuming pool size [2, 2]
    x_shape = x.get_shape().as_list()
    traits_num = x_shape[1] * x_shape[2] * filter_sizes[-1]

    # every polling layer decreases size by 4
    prev_input_size = traits_num // (4 ** len(filter_sizes))

    # change matrix to vector for feed-forward input
    current_layer = tf.reshape(current_layer, [-1, prev_input_size])

    # build feed-forward layers
    for deep_size, index in zip(deep_sizes, range(len(deep_sizes))):

        W = tf.get_variable(
            "W" + str(index),
            shape=[prev_input_size, deep_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        b = tf.get_variable(
            "b" + str(index),
            shape=[deep_size, ],
            initializer=tf.constant_initializer()
        )

        current_layer = tf.matmul(current_layer, W) + b
        prev_input_size = deep_size
        index += 1

    return tf.argmax(current_layer, axis=1), current_layer
