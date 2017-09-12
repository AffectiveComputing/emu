""" This module is responsible for model control. """


from os import path

import numpy as np
import tensorflow as tf

from model.net import net
from model.dataset import Dataset
from utils.dirs_utils import initialize_run_dirs


class Model(object):
    """
    A class that encapsulates all tensorflow graph functionality, necessary
    logging and saving graph to a file required for its training.

    Naming conventions:
    - all placeholders have prefix "in_"
    - values output by session evaluations have suffix "_out"
    - "val" is short for "validation"

    """

    # In and out shapes.
    INPUT_SIZE = (48, 48)
    IN_SHAPE = (None,) + INPUT_SIZE + (1,)
    CLASSES_COUNT = 7

    # Names for exported model's nodes.
    IN_DATA_NAME = "in_data"
    IN_FC_DROPOUT_NAME = "in_fc_dropout"
    IN_CONV_DROPOUT_NAME = "in_conv_dropout"
    SCORES_NAME = "scores"

    # Filenames for saved model's files.
    META_FILENAME = "model.meta"
    MODEL_FILENAME = "model"

    def __init__(self, net_architecture, data_root, run_name):
        (self._checkpoints_root, self._best_model_root, self._train_root,
         self._val_root) = initialize_run_dirs(data_root, run_name)
        self._checkpoint_file_path = path.join(self._checkpoints_root,
                                               self.MODEL_FILENAME)
        self._best_model_file_path = path.join(self._best_model_root,
                                               self.MODEL_FILENAME)
        self._create_inputs()
        self._create_output_nodes(net_architecture)
        self._create_training_nodes()
        self._create_summaries()
        self._create_environment()

    def train(
            self, dataset, learning_rate=0.001, desired_loss=0.001,
            max_epochs=1000000, decay_interval=10, decay_rate=1.0,
            batch_size=100, save_interval=1000, best_save_interval=200,
            validation_interval=200, fc_dropout=0.5, conv_dropout=0.25
    ):
        """ Public entry method for model's training. """
        self._session.run(tf.global_variables_initializer())
        min_loss = -np.log(1 / self.CLASSES_COUNT)
        for epoch in range(max_epochs):
            train_loss_out = self._do_train_run(
                epoch, dataset, batch_size, learning_rate, fc_dropout,
                conv_dropout
            )
            if epoch % validation_interval == 0:
                val_loss_out = self._do_val_run(epoch, dataset, batch_size,
                                                learning_rate)
            if epoch % decay_interval == 0:
                learning_rate *= decay_rate
            if epoch % save_interval == 0:
                self._checkpoints_saver.save(
                    self._session, self._checkpoint_file_path, global_step=epoch
                )
            # Save best model basing on validation loss.
            if epoch % best_save_interval == 0 and val_loss_out < min_loss:
                min_loss = val_loss_out
                self._best_model_saver.save(self._session,
                                            self._best_model_file_path)
            if train_loss_out < desired_loss:
                break

    def _do_train_run(self, epoch, dataset, batch_size, learning_rate,
                      fc_dropout, conv_dropout):
        """ Perform single train run. """
        data, labels = dataset.next_batch(batch_size, Dataset.TRAIN_I)
        _, loss_out, accuracy_out, train_summary_out = self._session.run(
            [self._train, self._loss, self._accuracy, self._train_summary],
            feed_dict={self._in_data: data, self._in_labels: labels,
                       self._in_learning_rate: learning_rate,
                       self._in_fc_dropout: fc_dropout,
                       self._in_conv_dropout: conv_dropout}
        )
        self._output_log_to_console("train", epoch, loss_out,
                                    accuracy_out, learning_rate)
        self._train_writer.add_summary(train_summary_out, global_step=epoch)
        return loss_out

    def _do_val_run(self, epoch, dataset, batch_size, learning_rate):
        """ Perform single validation run. """
        losses = []
        accuracies = []
        loops_count = (dataset.get_set_size(Dataset.VALIDATION_I) // batch_size)
        for i in range(loops_count):
            data, labels = dataset.next_batch(batch_size, Dataset.VALIDATION_I)
            loss_out, accuracy_out = self._session.run(
                [self._loss, self._accuracy],
                feed_dict={self._in_data: data, self._in_labels: labels,
                           self._in_fc_dropout: 0.0, self._in_conv_dropout: 0.0}
            )
            losses.append(loss_out)
            accuracies.append(accuracy_out)
        mean_loss = np.mean(losses)
        mean_accuracy = np.mean(accuracies)
        self._output_log_to_console("validation", epoch, mean_loss,
                                    mean_accuracy, learning_rate)
        val_summary_out, = self._session.run(
            [self._val_summary],
            feed_dict={self._in_loss: mean_loss,
                       self._in_accuracy: mean_accuracy}
        )
        self._val_writer.add_summary(val_summary_out, global_step=epoch)
        return mean_loss

    def _create_inputs(self):
        """ Initialize input placeholders used in training. """
        self._in_data = tf.placeholder(tf.float32, self.IN_SHAPE)
        tf.add_to_collection(self.IN_DATA_NAME, self._in_data)
        self._in_labels = tf.placeholder(tf.int64, [None, ])
        self._in_learning_rate = tf.placeholder(tf.float32, [])
        self._in_fc_dropout = tf.placeholder(tf.float32, [])
        tf.add_to_collection(self.IN_FC_DROPOUT_NAME, self._in_fc_dropout)
        self._in_conv_dropout = tf.placeholder(tf.float32, [])
        tf.add_to_collection(self.IN_CONV_DROPOUT_NAME, self._in_conv_dropout)

    def _create_output_nodes(self, net_architecture):
        """ Initialize model's output nodes. """
        self._output = net(self._in_data, net_architecture, self._in_fc_dropout,
                           self._in_conv_dropout)
        self._predictions = tf.argmax(self._output, axis=1)
        self._scores = tf.nn.softmax(self._output)
        tf.add_to_collection(self.SCORES_NAME, self._scores)

    def _create_training_nodes(self):
        """ Create training output nodes. """
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self._output,
                labels=tf.one_hot(self._in_labels, self.CLASSES_COUNT)
            )
        )
        self._accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(self._predictions, self._in_labels),
                dtype=tf.float32
            )
        )
        self._train = tf.train.AdamOptimizer(
            learning_rate=self._in_learning_rate
        ).minimize(self._loss)

    def _create_summaries(self):
        """ Create train and validation summary nodes. """
        train_loss_summary = tf.summary.scalar("loss", self._loss)
        train_accuracy_summary = tf.summary.scalar("accuracy", self._accuracy)
        self._train_summary = tf.summary.merge(
            [train_loss_summary, train_accuracy_summary]
        )
        # Validation summary needs dedicated placeholders for mean loss and
        # accuracy calculated during one full validation run.
        self._in_loss = tf.placeholder(tf.float32, [])
        self._in_accuracy = tf.placeholder(tf.float32, [])
        val_loss_summary = tf.summary.scalar("val_loss", self._in_loss)
        val_accuracy_summary = tf.summary.scalar("val_accuracy",
                                                 self._in_accuracy)
        self._val_summary = tf.summary.merge(
            [val_loss_summary, val_accuracy_summary]
        )

    def _create_environment(self):
        """ Create training environment. """
        self._session = tf.Session()
        self._best_model_saver = tf.train.Saver(max_to_keep=1)
        self._checkpoints_saver = tf.train.Saver(max_to_keep=10)
        self._train_writer = tf.summary.FileWriter(self._train_root)
        self._val_writer = tf.summary.FileWriter(self._val_root)

    @staticmethod
    def _output_log_to_console(run_type, epoch, loss, accuracy, learning_rate):
        """ Print training log to stdout. """
        print(
            "{} - Iteration {}:\n\tLoss: {}\n\tAccuracy: {}""\n\tLearning "
            "rate: {}".format(run_type, epoch, loss, accuracy, learning_rate)
        )
