from os import listdir
from os.path import isfile, join, basename, splitext

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

from const import WIDTH, HEIGHT

img_formats = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".tiff", ".TIFF", ".ico", ".ICO"]
img_255_formats = [".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp", ".BMP", ".tiff", ".TIFF", ".ico", ".ICO"]


# convert all images in source_dir into np arrays
# and export them into destination_dir

def export_as_np_array(source_dir="", destination_dir="", grayscale=False, scale=1.0):
    # for all files in source_dir
    for img_file in [join(source_dir, f) for f in listdir(source_dir) if is_img(join(source_dir, f))]:

        # convert image file to numpy array of fixed resolution
        image = img_to_np(img_file, grayscale, scale)

        # TODO this sucks
        # check for files with the same base names
        out_name = join(destination_dir, splitext(basename(img_file))[0])
        while isfile(out_name + ".npy"):
            print(out_name + ".npy appears more than once! Index '2' added.")
            out_name += "2"

        # export obtained np array and save in destination_dir in .npy format
        np.save(out_name, image)


# convert an image into np array

def img_to_np(img_file, grayscale=False, scale=1.0):
    # convert image file to numpy array
    image = mpimg.imread(img_file)

    if grayscale and image.ndim == 3:
        # get rid of alpha chanel
        image = image[..., :3]

        # convert pixels from [R, G, B] format to [luminance] format
        # using common [0.299, 0.587, 0.114] YUV factor
        image = np.dot(image, [0.299, 0.587, 0.114])

    # check if scale is needed in order to avoid computations
    if scale != 1.0:
        image = np.dot(image, scale)

    # change upper bound from 255 to 1. for JPG
    if splitext(img_file)[1] in img_255_formats:
        image *= 1. / 255

    return image


def resize_img(source_dir, destination_dir, width=WIDTH, height=HEIGHT):
    for filename in listdir(source_dir):
        img = Image.open(join(source_dir, filename))
        resized_image = img.resize((width, height), Image.ANTIALIAS)
        resized_image.save(join(destination_dir, filename))


# check if a file exists and whether it has a proper extension

def is_img(file_name):
    return isfile(file_name) and splitext(basename(file_name))[1] in img_formats
