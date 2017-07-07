from random import shuffle

import numpy as np


class DataSet(object):

    def __init__(self, src_dir, labels_file, shuffled = True):
        self.src_dir = src_dir

        with open(labels_file) as f:
            content = f.readlines()
            images = [x.strip().split()[0] for x in content]
            classes = [int(x.strip().split()[1]) for x in content]

            self.labels = list(zip(images, classes))

            if shuffled:
                shuffle(self.labels)

    def get_data(self, size, shape):
        if len(self.labels) == 0:
            return None, None

        data = np.empty((size,) + shape)
        labels = np.empty(size)

        i = 0
        while i < size:
            if len(self.labels) == 0:
                break

            label = self.labels.pop()
            img = np.load(self.src_dir + "/" + label[0])

            if img.shape == shape:
                data[i, :, :, :] = img
                labels[i] = label[1]
                i += 1

        return data, labels

    @property
    def size(self):
        return len(self.labels)
