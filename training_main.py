
from model.const import SOURCE_DIR, LABELS_FILE
from model.dataset import DataSet
from model.model import Model

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def main():
    model = Model()
    model.logs = True
    data_set = DataSet(SOURCE_DIR, LABELS_FILE)

    model.train(data_set, 0.003, 0.001, 100000, 5, 0.99, 100, 5, 20)


if __name__ == "__main__":
    main()
