import numpy as np
import pandas as pd
from imgaug import augmenters as iaa

points = [
    (2, 0, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 2, 0),
    (0, 0, 0, 2)
]

AUGMENTERS = [iaa.Crop(px=p) for p in points]
AUGMENTERS += [iaa.Sequential([
    iaa.Affine(scale={"x": 1.05, "y": 1.05}, rotate=(5, 0)),
    iaa.Crop(px=p)
]) for p in points]


def augment_skewed_class(data_frame):
    aug_images = []
    labels = []
    for image, label in zip(data_frame['pixels'], data_frame['emotion']):
        image = np.fromstring(image, sep=" ").reshape((48, 48))
        image = np.asarray(np.dstack((image, image, image)))
        for augmenter in AUGMENTERS:
            new_image = augmenter.augment_image(image)[:, :, 0]
            aug_images.append(new_image.reshape((48, 48, 1)))
            labels.append(label)

    return np.array(aug_images), np.array(labels)


def main():
    data = pd.read_csv("data/dataset/fer2013.csv")
    images, labels = augment_skewed_class(data)
    np.save('data/dataset/images.npy', images)
    np.save('data/dataset/labels.npy', labels)


if __name__ == "__main__":
    main()
