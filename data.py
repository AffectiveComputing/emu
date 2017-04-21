import os

import numpy as np


def get_data(source_dir, labels_file, max_size, shape):
    with open(labels_file) as f:
        content = f.readlines()
        keys = [x.strip().split()[0] for x in content]
        values = [int(x.strip().split()[1]) for x in content]
        labels_list = dict(zip(keys, values))

    labels = np.empty(max_size)
    data = np.empty((max_size,) + shape + (1,))

    i = 0
    for file in os.listdir(source_dir):
        if file.endswith(".npy"):
            img = np.load(source_dir + "/" + file)
            if img.shape == shape:
                data[i, :, :, 0] = img
                labels[i] = labels_list.get(file)
                i += 1

            if max_size <= i:
                break

    return data, labels
