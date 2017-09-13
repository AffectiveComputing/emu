""" This is an entry point for training of the model. """


from model.dataset import Dataset
from model.model import Model


def main():
    # Run configuration.
    layers = [{"type": "conv", "filters_count": 32, "kernel_size": 5},
              {"type": "conv", "filters_count": 64, "kernel_size": 5},
              {"type": "pool"},
              {"type": "conv", "filters_count": 128, "kernel_size": 5},
              {"type": "pool"},
              {"type": "conv", "filters_count": 128, "kernel_size": 5},
              {"type": "pool"},
              {"type": "conv", "filters_count": 256, "kernel_size": 5},
              {"type": "deep", "out_size": 100},
              {"type": "deep", "out_size": 7}]
    params = {"learning_rate": 0.001, "decay_rate": 1.0, "batch_size": 200,
              "conv_dropout": 0.35}
    model = Model(layers, "data", "run_15")
    data_set = Dataset("data/dataset/images.npy", "data/dataset/labels.npy")
    model.train(data_set, **params)


if __name__ == "__main__":
    main()
