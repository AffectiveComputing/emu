"""
This module contains definition of the data set object.
"""


import pickle
from os import path

import numpy as np

from preprocessing import image_utilities


__author__ = ["Michał Górecki", "Paweł Kopeć"]


class DataSet(object):
    """
    Object which handles train, validation and test subsets, shuffles the data
    and rewinds it after all the data has been served.
    """

    def __init__(self, source_path, labels_path, subsets_sizes=(0.8, 0.1, 0.1)):
        # Load labels dictionary.
        with open(labels_path, "rb") as labels_file:
            labels_dict = pickle.load(labels_file)
        images, labels = self.__load_data_set(labels_dict, source_path)
        self.__split_data_set(images, labels, subsets_sizes)
        # Local index of the currently server batch start.
        self.__batch_i = 0

    def next_train_batch(self, size):
        """
        Retrieve the next training batch with given size.
        :param size: size of the desired batch (in elements)
        :return: retrieved batch - both images and labels
        """
        train_set_size = self.__train_images.shape[0]
        # If there is not enough elements in the set for this batch,
        # reshuffle the whole set and reset batch index.
        if self.__batch_i + size > train_set_size:
            self.__reshuffle_train_set()
            self.__batch_i = 0
        images = self.__train_images[self.__batch_i:self.__batch_i + size]
        labels = self.__train_labels[self.__batch_i:self.__batch_i + size]
        self.__batch_i += size
        return images, labels

    def validation_batch(self):
        """
        Access the whole validation batch.
        :return: validation images and labels combined
        """
        return self.__validation_images, self.__validation_labels

    def test_batch(self):
        """
        Access the whole test batch.
        :return: test images and labels combined
        """
        return self.__test_images, self.__test_labels

    def __reshuffle_train_set(self):
        """
        Randomly change order of the train set to randomize batches.
        :return: -
        """
        # Cache current random generator state.
        current_random_state = np.random.get_state()
        np.random.shuffle(self.__train_images)
        # Restore random state from before images shuffling, so that the next
        # shuffling will be the same.
        np.random.set_state(current_random_state)
        np.random.shuffle(self.__train_labels)

    @staticmethod
    def __load_data_set(labels_dict, source_path):
        """
        Load all image files basing on the given labels dictionary and path to
        the data directory.
        :param labels_dict: loaded dictionary of labels for the image set
        :param source_path: path to the directory which contains image files
        :return: loaded images matrix and labels vector
        """
        images = []
        labels = []
        # Load image files from the disk with dict keys used as filenames.
        for key in labels_dict:
            image = np.load(path.join(source_path, key + ".npy"))
            if image_utilities.is_grayscale(image):
                image = np.expand_dims(image, -1)
            images.append(image)
            labels.append(labels_dict[key])
        images = np.array(images)
        labels = np.array(labels, dtype=np.int)
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
        train_indices = elements_order[:split_point_1]
        validation_indices = elements_order[split_point_1:split_point_2]
        test_indices = elements_order[split_point_2:]
        # Split the data set.
        self.__train_images = images[train_indices]
        self.__train_labels = labels[train_indices]
        self.__validation_images = images[validation_indices]
        self.__validation_labels = labels[validation_indices]
        self.__test_images = images[test_indices]
        self.__test_labels = labels[test_indices]
