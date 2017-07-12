"""
This module is responsible for model control.
"""


import numpy as np
import tensorflow as tf

from model.const import *
from model.net import net

__author__ = ["Paweł Kopeć", "Michał Górecki"]


class Model(object):

    def __init__(self, net_architecture):
        self.__prepared_for_training = False
        self.__in_data = tf.placeholder(tf.float32, IN_SHAPE)
        self.__create_output_nodes(net_architecture)
        self.__create_environment()

    def infer(self, x, model_dir):
        """

        :param x:           input placeholder
        :param model_dir:   directory with model files
        :return:
        """
        self.__session.run(tf.global_variables_initializer())
        self.__best_model_saver.restore(self.__session, model_dir)
        results = self.__session.run(
            self.__scores, feed_dict={self.__in_data: x}
        )
        return results

    def train(
        self, data_set, learning_rate, desired_loss, max_epochs,
            decay_interval, decay_rate, batch_size, save_interval,
            best_save_interval, validation_interval
    ):
        self.__prepare_for_training()
        self.__session.run(tf.global_variables_initializer())
        min_loss = -np.log(1 / CLASSES_COUNT)
        for epoch in range(max_epochs):
            # Obtain training data batch.
            data, labels = data_set.next_train_batch(batch_size)
            # Perform train run.
            _, current_loss, accuracy_value, summary_out = self.__session.run(
                [self.__train, self.__loss, self.__accuracy, self.__summary],
                feed_dict={
                    self.__in_data: data,
                    self.__in_labels: labels,
                    self.__in_learning_rate: learning_rate,
                }
            )
            # Output data to console and summary to log.
            output_log_to_console(
                "train", epoch, current_loss, accuracy_value, learning_rate
            )
            self.__train_writer.add_summary(summary_out, global_step=epoch)
            # Perform validation run every 'validation interval' steps.
            if epoch % validation_interval == 0:
                data, labels = data_set.validation_batch()
                current_loss, accuracy_value, summary_out = self.__session.run(
                    [self.__loss, self.__accuracy, self.__summary],
                    feed_dict={
                        self.__in_data: data,
                        self.__in_labels: labels,
                        self.__in_learning_rate: learning_rate,
                    }
                )
                output_log_to_console(
                    "validation", epoch, current_loss,
                    accuracy_value, learning_rate
                )
                self.__validation_writer.add_summary(
                    summary_out, global_step=epoch
                )
            # Decay the learning rate every 'decay_interval' steps.
            if epoch % decay_interval == 0:
                learning_rate *= decay_rate
            # Save model, if necessary conditions are met.
            if epoch % save_interval == 0:
                self.__checkpoints_saver.save(
                    self.__session, CHECKPOINTS_FILE, global_step=epoch
                )
            if epoch % best_save_interval == 0 and current_loss < min_loss:
                min_loss = current_loss
                self.__best_model_saver.save(
                    self.__session, MODEL_FILE
                )
            # Break, if the training target has been accomplished.
            if current_loss < desired_loss:
                break

    def __create_output_nodes(self, net_architecture):
        self.__output = net(self.__in_data, net_architecture)
        self.__predictions = tf.argmax(self.__output, axis=1)
        self.__scores = tf.nn.softmax(self.__output)

    def __create_environment(self):
        self.__session = tf.Session()
        self.__best_model_saver = tf.train.Saver(max_to_keep=1)

    def __prepare_for_training(self):
        if not self.__prepared_for_training:
            self.__in_labels = tf.placeholder(tf.int64, [None, ])
            self.__in_learning_rate = tf.placeholder(tf.float32, [])
            self.__create_training_nodes()
            self.__create_summary()
            self.__checkpoints_saver = tf.train.Saver(max_to_keep=10)
            self.__train_writer = tf.summary.FileWriter("data/logs/train")
            self.__validation_writer = tf.summary.FileWriter(
                "data/logs/validation"
            )
            self.__prepared_for_training = True

    def __create_training_nodes(self):
        self.__loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.__output,
                labels=tf.one_hot(self.__in_labels, CLASSES_COUNT)
            )
        )
        self.__accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(self.__predictions, self.__in_labels),
                dtype=tf.float32
            )
        )
        self.__train = tf.train.AdamOptimizer(
            learning_rate=self.__in_learning_rate
        ).minimize(self.__loss)

    def __create_summary(self):
        tf.summary.scalar("loss", self.__loss)
        tf.summary.scalar("accuracy", self.__accuracy)
        self.__summary = tf.summary.merge_all()


def output_log_to_console(run_type, epoch, loss, accuracy, learning_rate):
    print("{} - Iteration {}:".format(run_type, epoch))
    print("\tLoss ", loss)
    print("\tAccuracy", accuracy)
    print("\tLearning rate ", learning_rate)
