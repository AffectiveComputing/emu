"""
This script contains functions, which allow image transformation. It can
also serve as standalone command line utility, which can be used to
transform all files in a given directory.
"""


import cv2
import argparse
import numpy as np
from os import listdir, path


__author__ = ["Michał Górecki"]


def transform_image(image, grayscale, size):
    """
    Load and apply desired transformation to the input image.
    :param path: path to the image file
    :param grayscale: states whether to convert image to grayscale or to rgb
    :param size: desired size of the image
    :return: transformed image
    """
    # Constant count of grayscale/rgb image dimensions.
    GRAYSCALE_DIMENSION_COUNT = 2
    RGB_DIMENSION_COUNT = 3
    # Convert image to the required color space if needed.
    if len(image.shape) != GRAYSCALE_DIMENSION_COUNT and grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) != RGB_DIMENSION_COUNT and not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Resize the image.
    image = cv2.resize(image, tuple(size))
    # Scale the image to 0 - 1 range if its type is not floating point type.
    if not np.issubdtype(image.dtype, float):
        image =  image / np.iinfo(image.dtype).max
    return image


def transform_images_from_directory(in_path, out_path, grayscale, size):
    """
    Transform all image files contained in the input directory and save them
    under the same names in the output directory.
    :param in_path: path to the input directory
    :param out_path: path to the output directory
    :param grayscale: states whether to convert images to grayscale or to rgb
    :param size: desired size of the images
    :return: -
    """
    for filename in listdir(in_path):
        image = cv2.imread(path.join(in_path, filename))
        # Skip if image was unsuccessfully read.
        if image is None:
            continue
        # Transform the image.
        transformed_image = transform_image(image, grayscale, size)
        base_filename, _ = path.splitext(path.basename(filename))
        # Save transformed image matrix to a numpy file.
        np.save(path.join(out_path, base_filename), transformed_image)


def create_parser():
    """
    Create command line parser for this script.
    :return: created parser
    """
    parser = argparse.ArgumentParser(
        description="This script transforms input images "
    )
    image_colouring = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "in_path", type=str,
        help="Path to the input directory/file. In case of directory path "
             "given, this script attempts to convert all files from this "
             "directory."
    )
    parser.add_argument(
        "out_path", type=str,
        help="Path to the output directory, in which results of extraction "
             "are stored."
    )
    image_colouring.add_argument(
        "-gray", "--grayscale", action="store_true",
        help="Transform input images to grayscale. Exclusive with --rgb "
             "option."
    )
    image_colouring.add_argument(
        "-rgb", "--rgb", action="store_true",
        help="Transform input images to rgb format."
    )
    parser.add_argument(
        "-size", "--target_size", type=int,
        nargs=2, default=(128, 128),
        help="Specify target dimension of image transformation in width, "
             "height order."
    )
    return parser


def main():
    """
    Main function of this module.
    :return: -
    """
    # Parse command-line arguments.
    parser = create_parser()
    arguments = parser.parse_args()
    # Load cascade.
    transform_images_from_directory(
        arguments.in_path, arguments.out_path,
        arguments.grayscale, arguments.target_size
    )


if __name__ == "__main__":
    main()