"""
This module is responsible for model control.
"""


import numpy as np
import tensorflow as tf

from model.const import *
from model.net import Net

__author__ = ["Paweł Kopeć", "Michał Górecki"]


class Model(object):

    def __init__(self):
        self.logs = False

    def infer(self, x, meta_file, model_file):
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(session, model_file)

            scores = tf.get_collection("scores")[0]
            in_data = tf.get_collection("in_data")[0]
            results = session.run(scores, feed_dict={in_data: x})
            session.close()

            return results

    def train(
        self, data_set, learning_rate, desired_loss, max_epochs,
            decay_interval, decay_rate, batch_size, save_interval,
            best_save_interval
    ):
        in_data = tf.placeholder(tf.float32, IN_SHAPE)
        tf.add_to_collection("in_data", in_data)

        in_labels = tf.placeholder(tf.int64, [batch_size])
        in_learning_rate = tf.placeholder(tf.float32, [])

        #TODO make it not hard coded
        prediction, output, scores = Net.build_net(in_data, [32, 64], [CLASSES_NUM], 5)
        tf.add_to_collection("scores", scores)

        # Create loss, train, and accuracy nodes along with their summaries.
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=tf.one_hot(in_labels, CLASSES_NUM)
            )
        )
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(prediction, in_labels), dtype=tf.float32)
        )
        train = tf.train.AdamOptimizer(
            learning_rate=in_learning_rate
        ).minimize(loss)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        summary = tf.summary.merge_all()

        # Initialize session, summary writer and saver.
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("data/logs")
            checkpoints_saver = tf.train.Saver(max_to_keep=10)
            model_saver = tf.train.Saver(max_to_keep=1)

            batches = 0
            min_loss = -np.log(1 / CLASSES_NUM)

            while 0 < data_set.size:
                data, labels = data_set.get_data(batch_size, IMG_SHAPE)
                batches += 1

                for epoch in range(max_epochs):
                    # Perform train run.
                    _, current_loss, accuracy_value, summary_out = session.run(
                        [train, loss, accuracy, summary],
                        feed_dict={
                            in_data: data,
                            in_labels: labels,
                            in_learning_rate: learning_rate,
                        })

                    # Output data to console and summary to log.
                    if self.logs:
                        print("Iteration ", epoch)
                        print("\tBatch ", batches)
                        print("\tLoss ", current_loss)
                        print("\tAccuracy", accuracy_value)
                        print("\tLearning rate ", learning_rate)

                    writer.add_summary(summary_out, global_step=epoch)

                    if epoch % decay_interval == 0:
                        learning_rate *= decay_rate

                    if current_loss < desired_loss:
                        break

                    if epoch % save_interval == 0:
                        checkpoints_saver.save(
                            session, CHECKPOINTS_FILE, global_step=epoch
                        )

                    if epoch % best_save_interval == 0 and current_loss < min_loss:
                        min_loss = current_loss
                        model_saver.save(
                            session, MODEL_FILE
                        )

            session.close()