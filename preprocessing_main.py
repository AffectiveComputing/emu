from preprocessing.data_set_preparing import prepare_data_set

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def main():
    prepare_data_set(
        "data/png", "data/npy", "data/png/emotion.txt",
        apply_noise=True, apply_flip=True
    )

if __name__ == "__main__":
    main()
