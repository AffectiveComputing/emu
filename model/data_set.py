import numpy as np
import pandas as pd

__author__ = ["Michał Górecki", "Paweł Kopeć"]


class DataSet(object):
    # TODO handling errors

    """
    Class which handles serving, shuffling and rewinding train, validation and
    test subsets of data set.
    """

    TRAIN_I = 0
    VALIDATION_I = 1
    TEST_I = 2

    def __init__(self, csv_path, subsets_sizes=(0.8, 0.1, 0.1)):
        # Load labels dictionary.
        images, labels = self.__load_data_set(csv_path)
        self.__split_data_set(images, labels, subsets_sizes)
        # Local index of the currently server batch start.
        self.__batches_is = [0, 0, 0]

    def next_batch(self, size, set_i):
        """
        Retrieve the next training batch with given size.
        :param size: size of the desired batch (in elements)
        :return: retrieved batch - both images and labels
        """
        set_size = self.__images[set_i].shape[0]
        if self.__batches_is[set_i] + size > set_size:
            self.__reshuffle_set(set_i)
            self.__batches_is[set_i] = 0
        images = self.__images[set_i][
            self.__batches_is[set_i]:self.__batches_is[set_i] + size
        ]
        labels = self.__labels[set_i][
            self.__batches_is[set_i]:self.__batches_is[set_i] + size
        ]
        self.__batches_is[set_i] += size
        return images, labels

    def __reshuffle_set(self, set_i):
        """
        Randomly change order of the train set to randomize batches.
        :return: -
        """
        # Cache current random generator state.
        current_random_state = np.random.get_state()
        np.random.shuffle(self.__images[set_i])
        # Restore random state from before images shuffling, so that the next
        # shuffling will be the same.
        np.random.set_state(current_random_state)
        np.random.shuffle(self.__labels[set_i])

    @staticmethod
    def __load_data_set(csv_path):
        """
        Load all image files basing on the given labels dictionary and path to
        the data directory.

        :return: loaded images matrix and labels vector
        """
        images = np.load('data/dataset/images.npy')
        labels = np.load('data/dataset/labels.npy')

        return images, labels

    def __split_data_set(self, images, labels, subsets_sizes):
        """
        Split data set for train, validation and test subsets and initialize
        corresponding instance fields with those subsets.
        :param images: matrix of all images from the data set
        :param labels: vector of all labels for loaded images
        :param subsets_sizes: tuple of 3 float values which describe fraction
                              sizes of consecutively train, validation and
                              test subsets. It should sum to 1.0
        :return: -
        """
        elements_count = images.shape[0]
        # Unpack size values.
        train_size, validation_size, test_size = subsets_sizes
        # Determine data set split points.
        split_point_1 = int(train_size * elements_count)
        split_point_2 = split_point_1 + int(validation_size * elements_count)
        # Create one, random elements order and extract subsets indices from
        # it.
        elements_order = np.random.permutation(elements_count)
        #train_indices = elements_order[:split_point_1]
        #validation_indices = elements_order[split_point_1:split_point_2]
        #test_indices = elements_order[split_point_2:]
        all_indices = list(range(elements_count))
        train_indices = all_indices[:split_point_1]
        validation_indices = all_indices[split_point_1:split_point_2]
        test_indices = all_indices[split_point_2:]
        # Split the data set.
        self.__images = []
        self.__labels = []
        for indices in [train_indices, validation_indices, test_indices]:
            self.__images.append(images[indices])
            self.__labels.append(labels[indices])
