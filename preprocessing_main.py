"""
Main script of this package, which is used to prepare images for conv net
input.
"""


import numpy as np
from os import path, listdir

from preprocessing import image_utilities
from preprocessing import labels_handling
from preprocessing import parser


__author__ = ["Michał Górecki"]


def main():
    """
    Main function of this script.
    :return: -
    """
    # Parse command line arguments.
    arguments_parser = parser.create_parser()
    arguments = arguments_parser.parse_args()
    # Load cascade object and labels dictionary.
    cascade = image_utilities.load_cascade(arguments.cascade_path)
    labels = labels_handling.load_labels(arguments.labels_path)
    for filename in listdir(arguments.in_path):
        # Load next image.
        image = image_utilities.load_image(
            path.join(arguments.in_path, filename)
        )
        # Skip loop iteration if image failed to load.
        if image is None:
            continue
        # Process it.
        transformed_images, new_images = image_utilities.process_image(
            cascade, image, arguments.scale_factor, arguments.min_neighbours,
            arguments.apply_flip, arguments.apply_noise,
            arguments.noise_intensity, arguments.grayscale, arguments.rgb,
            arguments.target_size
        )
        # Extract base filename and create list of the new filenames.
        base_filename, extension = path.splitext(filename)
        new_filenames = [
            base_filename + "_{}".format(i) for i in range(len(new_images))
        ]
        label = labels[base_filename]
        # Save transformed numpy matrices and result images for verification
        # purposes.
        for transformed_image, new_image, new_filename in \
                zip(transformed_images, new_images, new_filenames):
            # Save transformed image and store its new label.
            np.save(
                path.join(arguments.out_path, new_filename), transformed_image
            )
            labels[new_filename] = label
            # Save image copy of transformation.
            image_utilities.save_image(
                path.join(arguments.out_path, new_filename + extension),
                new_image
            )
    # Save the new labels dictionary in output directory.
    labels_base_filename, _ = path.splitext(
        path.basename(arguments.labels_path)
    )
    labels_handling.save_labels(
        path.join(arguments.out_path, labels_base_filename), labels
    )


if __name__ == "__main__":
    main()