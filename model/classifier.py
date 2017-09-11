""" This module contains infer capable class. """


from os import path

import tensorflow as tf

from model.model import Model
from utils.dirs_utils import LOGS_ROOT, BEST_MODEL_DIR


class Classifier:
    """
    A class for inferring the class without initializing those graph nodes
    that are needed only for training.
    """

    def __init__(self, data_root, run_name):
        """ Setup environment. """
        best_model_root = path.join(
            path.join(path.join(data_root, LOGS_ROOT), run_name), BEST_MODEL_DIR
        )
        meta_file_path = path.join(best_model_root, Model.META_FILENAME)
        model_file_path = path.join(best_model_root, Model.MODEL_FILENAME)
        self._session = tf.Session()
        saver = tf.train.import_meta_graph(meta_file_path)
        saver.restore(self._session, model_file_path)
        self._in_data = tf.get_collection(Model.IN_DATA_NAME)[0]
        self._scores = tf.get_collection(Model.SCORES_NAME)[0]

    def infer(self, x):
        """ Feed froward x through the loaded net. """
        return self._session.run(self._scores, feed_dict={self._in_data: x})

    def close(self):
        self._session.close()
