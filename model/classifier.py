import tensorflow as tf


__author__ = ["Michał Górecki", "Paweł Kopeć"]


class Classifier:
    """
    A class for inferring the class without initializing those graph nodes
    that are needed only for training.
    """

    INPUT_SIZE = (48, 48)

    def __init__(self, model_path, meta_path):
        # Setup environment.
        self.__session = tf.Session()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.__session, model_path)
        self.__in_data = tf.get_collection("in_data")[0]
        self.__scores = tf.get_collection("scores")[0]

    def infer(self, x):
        """

        :param x:           input placeholder
        :param model_path:   directory with model files
        :return:
        """
        return self.__session.run(
            self.__scores, feed_dict={self.__in_data: x}
        )

    def close(self):
        self.__session.close()
