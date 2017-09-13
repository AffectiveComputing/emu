""" This module contains infer capable class. """

import tensorflow as tf


class Classifier:
    """
    A class for inferring the class without initializing those graph nodes
    that are needed only for training.
    """

    def __init__(self, meta_path, model_path):
        """ Read neural net model from file. """
        self._session = tf.Session()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self._session, model_path)
        self._in_data = tf.get_collection("in_data")[0]
        self._in_fc_dropout = tf.get_collection("in_fc_dropout")[0]
        self._in_conv_dropout = tf.get_collection("in_conv_dropout")[0]
        self._scores = tf.get_collection("scores")[0]

    def infer(self, x):
        """ Predict emotion. """
        return self._session.run(
            self._scores,
            feed_dict={self._in_data: x, self._in_fc_dropout: 0.0,
                       self._in_conv_dropout: 0.0}
        )

    def close(self):
        self._session.close()
