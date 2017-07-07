from os import listdir
from os.path import isfile, join, basename, splitext

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

img_formats = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG",
               ".bmp", ".BMP", ".tiff", ".TIFF", ".ico", ".ICO"]
img_255_formats = [".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp",
                   ".BMP", ".tiff", ".TIFF", ".ico", ".ICO"]


def export_as_np_in_dir(src_dir, out_dir, gray_scale=False, scale=1.0):
    """
    Convert all images in src_dir into np arrays
    and save them in out_dir.

    :param src_dir: a directory with images
    :param out_dir: a directory where numpy arrays will be written to
    :param gray_scale: True if images are to be converted to gray-scale
    :param scale: scale by which pixel value must be multiplied
    """

    # for all files in source_dir
    for img_file in [join(src_dir, f)
                     for f in listdir(src_dir) if is_img(join(src_dir, f))]:

        # convert image file to numpy array of fixed resolution
        image = img_to_np(img_file, gray_scale, scale)

        out_name = join(out_dir, splitext(basename(img_file))[0])

        # export obtained np array and save in destination_dir in .npy format
        np.save(out_name, image)


def img_to_np(path, gray_scale=False, scale=1.0):
    """
    Convert an image into np array.

    :param path: path to an img
    :param gray_scale: True if image is to be converted to gray-scale
    :param scale: scale by which pixel value must be multiplied
    :return: numpy array representing image
    """

    # convert image file to numpy array
    img_as_np = mpimg.imread(path)

    if gray_scale and img_as_np.ndim == 3:
        # get rid of alpha chanel
        img_as_np = img_as_np[..., :3]

        # convert pixels from [R, G, B] format to [luminance] format
        # using common [0.299, 0.587, 0.114] YUV factor
        img_as_np = np.dot(img_as_np, [0.299, 0.587, 0.114])

    # check if scaling is needed
    if scale != 1.0:
        img_as_np = np.dot(img_as_np, scale)

    # change upper bound from 255 to 1. for JPG
    if splitext(path)[1] in img_255_formats:
        img_as_np *= 1. / 255

    return img_as_np


def resize_img_in_dir(src_dir, dest_dir, width, height):
    for file in listdir(src_dir):
        if is_img(join(src_dir, file)):
            img = Image.open(join(src_dir, file))
            resized_image = resize_img(img, width, height)
            resized_image.save(join(dest_dir, file))


def resize_img(img, width, height):
    return img.resize((width, height), Image.ANTIALIAS)


def is_img(file_name):
    """
    Check if a file is an image.

    :param file_name: of a checked file
    :return: True if file exists and it has a proper image extension
    """
    return isfile(file_name) and splitext(basename(file_name))[1] in img_formats
