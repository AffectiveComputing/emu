
from model.const import SOURCE_DIR, LABELS_FILE
from model.data_set import DataSet
from model.model import Model

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def main():
    model = Model()
    model.logs = True
    data_set = DataSet(SOURCE_DIR, LABELS_FILE)

    layers = [
        {
            "type": "conv",
            "filters_count": 32,
            "kernel_size": 5,
        },
        {
            "type": "pool"
        },
        {
            "type": "conv",
            "filters_count": 128,
            "kernel_size": 5,
        },
        {
            "type": "deep",
            "out_size": 100
        },
        {
            "type": "deep",
            "out_size": 7
        }
    ]

    model.train(data_set, 0.003, 0.001, 100000, 5, 0.99, 100, 5, 20, layers)


if __name__ == "__main__":
    main()
