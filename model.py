"""
This module is responsible for model control.
"""


import tensorflow as tf
import numpy as np

from const import *
from net import Net


__author__ = ["Paweł Kopeć", "Michał Górecki"]


class Model(object):
    """

    """

    def __init__(self):
        self.logs = False

    def infer(self, x):
        session = tf.Session()
        saver = tf.train.import_meta_graph("best_model/model.meta")
        saver.restore(session, "best_model/model")
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        print(tf.get_collection(tf.GraphKeys.WEIGHTS))
        scores = tf.get_collection("scores")[0]
        in_data = tf.get_collection("in_data")[0]
        out_scores, = session.run(scores, feed_dict={in_data: x})
        return out_scores

    def train(
        self, data_set, learning_rate, desired_loss, max_epochs,
        decay_interval, decay_rate, batch_data_size
    ):
        """

        :param data_set:
        :param learning_rate:
        :param desired_loss:
        :param max_epochs:
        :param decay_interval:
        :param decay_rate:
        :param batch_data_size:
        :return:
        """
        CHECKPOINT_INTERVAL = 10
        BEST_MODEL_INTERVAL = 5
        # Set up model placeholders.
        in_data = tf.placeholder(tf.float32, IN_SHAPE)
        tf.add_to_collection("in_data", in_data)
        in_labels = tf.placeholder(tf.int64, [batch_data_size])
        in_learning_rate = tf.placeholder(tf.float32, [])
        #TODO make it not hard coded
        prediction, output = Net.build_net(in_data, [32, 64], [OUT_SIZE], 5)
        scores = tf.nn.softmax(output)
        tf.add_to_collection("scores", scores)
        # Create loss, train, and accuracy nodes along with their summaries.
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=tf.one_hot(in_labels, OUT_SIZE)
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
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs")
        checkpoints_saver = tf.train.Saver(max_to_keep=10)
        best_model_saver = tf.train.Saver(max_to_keep=1)
        batches = 0
        min_loss = -np.log(1 / OUT_SIZE)
        while 0 < data_set.size:
            data, labels = data_set.get_data(batch_data_size, IMG_SHAPE)
            batches += 1
            for i in range(max_epochs):
                # Perform train run.
                _, current_loss, accuracy_value, summary_out = session.run(
                    [train, loss, accuracy, summary],
                    feed_dict={
                        in_data: data,
                        in_labels: labels,
                        in_learning_rate: learning_rate,
                    })
                # Output data to console and summary to log.
                print("Iteration ", i)
                print("\tBatch ", batches)
                print("\tLoss ", current_loss)
                print("\tAccuracy", accuracy_value)
                print("\tLearning rate ", learning_rate)
                writer.add_summary(summary_out, global_step=i)
                # Decay learning rate and stop if learning target is
                # accomplished.
                if i % decay_interval == 0:
                    learning_rate *= decay_rate
                if current_loss < desired_loss:
                    break
                if i % CHECKPOINT_INTERVAL == 0:
                    checkpoints_saver.save(
                        session, "checkpoints/model", global_step=i
                    )
                if i % BEST_MODEL_INTERVAL == 0 and current_loss < min_loss:
                    min_loss = current_loss
                    best_model_saver.save(
                        session, "best_model/model"
                    )
        session.close()

