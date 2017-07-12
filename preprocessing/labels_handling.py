"""
This module is responsible for loading and saving the labels file.
"""


import pickle
from os import path


__author__ = ["Michał Górecki"]


def load_labels(in_path):
    """
    Load labels from txt file and store them in dictionary with base filenames
    as keys.
    :param in_path: path to the labels file
    :return: loaded labels dictionary
    """
    # Constant separator of filename, label pair used in txt file.
    SEPARATOR = " "
    labels = dict()
    # Open labels file and process all lines from it.
    labels_file = open(in_path, "r")
    for line in labels_file:
        filename, label = line.split(SEPARATOR)
        key, _ = path.splitext(filename)
        labels[key] = int(label)
    return labels


def save_labels(out_path, labels_dict):
    """
    Save labels dictionary to pickle file.
    :param labels_dict: dictionary of labels for filenames
    :return: -
    """
    labels_file = open(out_path, "wb")
    pickle.dump(labels_dict, labels_file)
    labels_file.close()
