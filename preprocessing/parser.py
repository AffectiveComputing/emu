"""
This module contains functions, which allow the main script to create its own
command line argument parser.
"""


import argparse


__author__ = ["Michał Górecki"]


def add_in_out_group(parser):
    """
    Add in and out path arguments to the parser.
    :param parser: modified parser
    :return: -
    """
    in_out_group = parser.add_argument_group(
        "In/out paths",
        "Arguments, which specify script input and output source."
    )
    in_out_group.add_argument(
        "in_path", type=str,
        help="Path to the input directory. Script will attempt to process all "
             "files contained in this directory. Non-image files will be "
             "skipped."
    )
    in_out_group.add_argument(
        "out_path", type=str,
        help="Path to the output directory, in which results of the script's "
             "execution will be stored."
    )
    in_out_group.add_argument(
        "labels_path", type=str,
        help="Path to the input labels file."
    )


def add_detection_group(parser):
    """
    Add arguments group related to face detection to the parser.
    :param parser: modified parser
    :return: -
    """
    detection_group = parser.add_argument_group(
        "Face detection", "Arguments used as parameters in face detection."
    )
    detection_group.add_argument(
        "-c_path", "--cascade_path",
        type=str, default="data/cascades/haarcascade_frontalface_default.xml",
        help="Path to the used xml cascade file."
    )
    detection_group.add_argument(
        "-s_fac", "--scale_factor",
        type=float, default=1.05,
        help="Factor used in image rescaling during face detection. It "
             "specifies how much the analyzed image will be rescaled with "
             "each detection step."
    )
    detection_group.add_argument(
        "-min_neigh", "--min_neighbours",
        type=int, default=5,
        help=""
    )


def add_augmentation_group(parser):
    """
    Add group of arguments responsible for data augmentation.
    :param parser: modified parser
    :return: -
    """
    augmentation_group = parser.add_argument_group(
        "Data augmentation",
        "These arguments are used in data augmentation process."
    )
    augmentation_group.add_argument(
        "-app_f", "--apply_flip",
        action="store_true",
        help="Indicate that the script should also save flipped copy of each "
             "image."
    )
    augmentation_group.add_argument(
        "-app_n", "--apply_noise",
        action="store_true",
        help="Activate noise augmentation. Original image will be copied with "
             "noise added to it."
    )
    augmentation_group.add_argument(
        "-n_int", "--noise_intensity",
        type=float, default=0.2,
        help="Intensity value of the used noise matrix. It should be a "
             "fraction (0.0 - 1.0) of image max color value."
    )


def add_transformation_group(parser):
    """
    Add arguments responsible for images transformation.
    :param parser: modified parser
    :return: -
    """
    transformation_group = parser.add_argument_group(
        "Image transformation",
        ""
    )
    image_colouring = transformation_group.add_mutually_exclusive_group()
    image_colouring.add_argument(
        "-gray", "--grayscale", action="store_true",
        help="Transform input images to grayscale. Exclusive with --rgb "
             "option."
    )
    image_colouring.add_argument(
        "-rgb", "--rgb", action="store_true",
        help="Transform input images to rgb format."
    )
    transformation_group.add_argument(
        "-size", "--target_size", type=int,
        nargs=2, default=(128, 128),
        help="Specify target dimension of image transformation in width, "
             "height order."
    )


def create_parser():
    """
    Create command line argument parser for main script.
    :return: created parser
    """
    parser = argparse.ArgumentParser(
        description="This script allows preparation of a set of images to "
                    "use them as input for conv net."
    )
    # Add argument groups to the parser.
    add_in_out_group(parser)
    add_detection_group(parser)
    add_augmentation_group(parser)
    add_transformation_group(parser)
    return parser