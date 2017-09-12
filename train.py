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
              {"type": "conv", "filters_count": 256, "kernel_size": 5},
              {"type": "pool"},
              {"type": "conv", "filters_count": 512, "kernel_size": 5},
              {"type": "deep", "out_size": 100},
              {"type": "deep", "out_size": 7}]
    params = {"learning_rate": 0.001, "decay_rate": 1.0, "batch_size": 200,
              "conv_dropout": 0.35}
    data_root = "data"
    run_name = "run_14"
    dataset_name = "default"
    model = Model(layers, data_root, run_name)
    data_set = Dataset(data_root, dataset_name)
    model.train(data_set, **params)


if __name__ == "__main__":
    main()
