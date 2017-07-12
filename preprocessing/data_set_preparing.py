from os import listdir, path, makedirs

import numpy as np

from preprocessing.image_utilities import load_cascade, load_image, \
    save_image, extract_faces, augment_images, convert_to_colorspace, resize, \
    is_float, normalize
from preprocessing.labels_handling import load_labels, save_labels

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def prepare_data_set(
        in_dir, out_dir, labels_file, grayscale=True, rgb=False,
        target_size=(128, 128),
        cascade_path="data/cascades/haarcascade_frontalface_default.xml",
        min_neighbours=5, scale_factor=1.05, apply_noise=False,
        apply_flip=False, noise_intensity=0.2
):
    if not path.exists(out_dir):
        makedirs(out_dir)
    cascade = load_cascade(cascade_path)
    labels = load_labels(labels_file)
    for filename in listdir(in_dir):
        # Load next image.
        image = load_image(
            path.join(in_dir, filename)
        )
        # Skip loop iteration if image failed to load.
        if image is None:
            continue
        transformed_images, new_images = process_image(
            cascade, image, scale_factor, min_neighbours,
            apply_flip, apply_noise,
            noise_intensity, grayscale, rgb,
            target_size
        )
        base_filename, extension = path.splitext(filename)
        new_filenames = [
            base_filename + "_{}".format(i) for i in range(len(new_images))
        ]
        label = labels[base_filename]
        labels.pop(base_filename, None)
        # Save transformed numpy matrices and result images for verification
        # purposes.
        for transformed_image, new_image, new_filename in \
                zip(transformed_images, new_images, new_filenames):
            # Save transformed image and store its new label.
            np.save(
                path.join(out_dir, new_filename), transformed_image
            )
            labels[new_filename] = label
            # Save image copy of transformation.
            save_image(
                path.join(out_dir, new_filename + extension),
                new_image
            )
    # Save the new labels dictionary in output directory.
    labels_base_filename, _ = path.splitext(
        path.basename(labels_file)
    )
    save_labels(
        path.join(out_dir, labels_base_filename), labels
    )


def process_image(
        cascade, in_image, scale_factor, min_neighbours,
        apply_flip, apply_noise, noise_intensity, grayscale, rgb, target_size
):
    """
    This function processes image with face detection, augmentation and
    transformation.
    :param cascade: cascade object used in face detection
    :param in_image: processed image
    :param scale_factor: scaling factor parameter for face detection
    :param min_neighbours: minimum neighbours parameter for face detection
    :param apply_flip: whether to apply flip augmentation
    :param apply_noise: whether to apply noise augmentation
    :param noise_intensity: intensity of the applied noise (0.0 - 1.0)
    :param grayscale: whether to convert images to grayscale
    :param rgb: whether to convert images to rgb color space
    :param target_size: desired size of transformed image
    :return: list of images before and after transformation
    """
    faces = extract_faces(cascade, in_image, scale_factor, min_neighbours)
    augmented_images = augment_images(
        faces, apply_flip, apply_noise, noise_intensity
    )
    converted_images = convert_to_colorspace(augmented_images, grayscale, rgb)
    resized_images = [resize(image, target_size) for image in converted_images]
    normalized_images = [
        image if is_float(image) else normalize(image)
        for image in resized_images
    ]
    return normalized_images, augmented_images
