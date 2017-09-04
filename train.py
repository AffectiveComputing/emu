from model.data_set import DataSet
from model.model import Model, DATA_SET_DIR

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def main():
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
            "type": "pool"
        },
        {
            "type": "conv",
            "filters_count": 256,
            "kernel_size": 5,
        },
        {
            "type": "pool"
        },
        {
            "type": "conv",
            "filters_count": 512,
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
    model = Model(layers)
    model.logs = True
    data_set = DataSet(DATA_SET_DIR)
    model.train(data_set, learning_rate=0.001, decay_rate=1.0, batch_size=100)


if __name__ == "__main__":
    main()
