import tensorflow as tf

from const import *
from net import Net


class Model(object):

    def __init__(self):
        self.logs = False

    def infer(self, x):
        #TODO
        pass

    def train(self, data_set, learning_rate, desired_loss, max_epochs,
              decay_interval, decay_rate, batch_data_size):

        x = tf.placeholder(tf.float32, IN_SHAPE)

        #TODO make it not hard coded
        prediction, output = Net.build_net(x, [32, 64], [OUT_SIZE], 5)

        correct = tf.placeholder(tf.int64, [batch_data_size])

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
                data, labels = data_set.get_data(batch_data_size, IMG_SHAPE)
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
