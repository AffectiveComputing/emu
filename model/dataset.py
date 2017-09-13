""" This module contains main Dataset class. """

import numpy as np


class Dataset(object):
    """
    Class which handles serving, shuffling and rewinding train, validation and
    test subsets of data set.
    """

    # TODO handling errors

    # Indices of subsets in internal sets list.
    TRAIN_I = 0
    VALIDATION_I = 1
    TEST_I = 2

    def __init__(self, images_path, labels_path, subsets_sizes=(0.8, 0.1, 0.1)):
        images = np.load(images_path)
        labels = np.load(labels_path)
        self._split_dataset(images, labels, subsets_sizes)
        self._batches_is = [0, 0, 0]

    def next_batch(self, size, set_i):
        """
        Get next batch of images and labels with given size from chosen
        set.
        """
        set_size = self.get_set_size(set_i)
        # Reshuffle whole set and start batching from the beginning, if more
        # images are requested than left in the set.
        if self._batches_is[set_i] + size > set_size:
            self._reshuffle_set(set_i)
            self._batches_is[set_i] = 0
        images = self._images[set_i][
                 self._batches_is[set_i]:self._batches_is[set_i] + size
                 ]
        labels = self._labels[set_i][
                 self._batches_is[set_i]:self._batches_is[set_i] + size
                 ]
        self._batches_is[set_i] += size
        return images, labels

    def get_set_size(self, set_i):
        """ Return size in elements of selected set. """
        return self._images[set_i].shape[0]

    def _reshuffle_set(self, set_i):
        """ Randomly change order of the chosen set to reshuffle batches. """
        current_random_state = np.random.get_state()
        np.random.shuffle(self._images[set_i])
        np.random.set_state(current_random_state)
        np.random.shuffle(self._labels[set_i])

    def _split_dataset(self, images, labels, subsets_sizes):
        """
        Split data set for train, validation and test subsets and initialize
        corresponding instance fields with those subsets.
        """
        elements_count = images.shape[0]
        train_size, validation_size, test_size = subsets_sizes
        split_point_1 = int(train_size * elements_count)
        split_point_2 = split_point_1 + int(validation_size * elements_count)
        # Since current data set augmentation doesn't fool our CNN, random
        # ordering of the examples for split must be ditched.
        # elements_order = np.random.permutation(elements_count)
        elements_order = list(range(elements_count))
        train_indices = elements_order[:split_point_1]
        validation_indices = elements_order[split_point_1:split_point_2]
        test_indices = elements_order[split_point_2:]
        self._images = []
        self._labels = []
        for indices in [train_indices, validation_indices, test_indices]:
            self._images.append(images[indices])
            self._labels.append(labels[indices])
