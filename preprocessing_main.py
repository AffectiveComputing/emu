from os import path, listdir

from preprocessing.data_set_preparing import prepare_data_set

__author__ = ["Paweł Kopeć", "Michał Górecki"]


DATA_DIR = "data/png"
NUMPY_DIR = "data/npy"
LABELS_FILE = "data/png/labels.txt"


def main():
    if len(listdir(DATA_DIR)) is 2:
        print("First download database from",
              "http://www.consortium.ri.cmu.edu/ckagree/",
              "and put it into", DATA_DIR, "folder")
        return

    prepare_data_set(
        DATA_DIR, NUMPY_DIR, LABELS_FILE,
        apply_noise=True, apply_flip=True
    )

if __name__ == "__main__":
    main()
