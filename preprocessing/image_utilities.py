"""
This module contains image transformations, conversions and checks. It also
contains face detection and extraction routines.
"""


import cv2
import numpy as np


__author__ = ["Michał Górecki"]


def load_image(path):
    """
    Load image from the file on the disk.
    :param path: path to the input file
    :return: loaded image or false if it fails
    """
    return cv2.imread(path)


def save_image(path, image):
    """
    Save input image on the disk.
    :param path: path to the output file
    :param image: input image
    :return: -
    """
    cv2.imwrite(path, image)


def crop(image, rectangle):
    """
    Crop out input image's fragment specified by the rectangle.
    :param image: input image
    :param rectangle: rectangle, which indicates cropped area
    :return: cropped original image fragment
    """
    x, y, w, h = rectangle
    return image[y:y + h, x:x + w]


def resize(image, size):
    """
    Resize given image.
    :param image: input image
    :param size: desired new image size (tuple of width, height)
    :return: resized image
    """
    return cv2.resize(image, size)


def flip(image):
    """
    Flip given image horizontally.
    :param image: input image
    :return: flipped image
    """
    HORIZONTAL_FLIP_FLAG = 1
    return cv2.flip(image, HORIZONTAL_FLIP_FLAG)


def is_float(image):
    """
    Check if image's color values are float values.
    :param image: checked image
    :return: result of the check
    """
    return np.issubdtype(image.dtype, float)


def normalize(image):
    """
    Normalize image colors to 0.0 - 1.0 range.
    :param image: input image with colors in integer space
    :return: normalized image
    """
    return image / np.iinfo(image.dtype).max


def add_noise(image, intensity):
    """
    Add noise mask to the image.
    :param image: input image
    :param intensity: fraction, which describes how much of the image max
                      allowed color value, should be the noise limit
    :return: image with noise applied to it
    """
    # Determine max color value of the input image's format and max noise.
    if is_float(image):
        max_color = 1.0
    else:
        max_color = np.iinfo(image.dtype).max
    max_noise = intensity * max_color
    # Create noise with uniform distribution and apply it.
    noise = np.random.uniform(-max_noise, max_noise, image.shape)
    noise_image = image + noise
    return np.minimum(np.maximum(0, noise_image), max_color).astype(image.dtype)


def is_grayscale(image):
    """
    Check if given image is in grayscale color space.
    :param image: checked image
    :return: result of the check (boolean value)
    """
    GRAYSCALE_DIMENSION_COUNT = 2
    return len(image.shape) == GRAYSCALE_DIMENSION_COUNT


def is_rgb(image):
    """
    Check if given image is in RGB color space.
    :param image: checked image
    :return: result of the check (boolean value)
    """
    RGB_DIMENSION_COUNT = 3
    return len(image.shape) == RGB_DIMENSION_COUNT


def to_grayscale(image):
    """
    Convert image to the grayscale color space.
    :param image: input image (numpy matrix)
    :return: converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_rgb(image):
    """
    Convert image to the RGB color space.
    :param image: input image (numpy matrix)
    :return: converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def load_cascade(path):
    """
    Load opencv haar cascade object from xml file.
    :param path: path to the xml cascade file
    :return: loaded cascade object
    """
    return cv2.CascadeClassifier(path)


def detect_faces(cascade, image, scale_factor, min_neighbours):
    """
    Detect faces visible on the image.
    :param cascade: cascade object used for detection
    :param image: analyzed image
    :param scale_factor: subsequent detections scaling coefficient
    :param min_neighbours: minimum detection neighbours
    :return: detected face rectangles
    """
    return cascade.detectMultiScale(image, scale_factor, min_neighbours)


def extract_faces(cascade, image, scale_factor, min_neighbours):
    """
    Extract faces present on the input image.
    :param cascade: cascade object used for detection
    :param image: input image
    :param scale_factor: subsequent detections scaling coefficient
    :param min_neighbours: minimum detection neighbours
    :return: image parts with faces detected on them
    """
    # Convert image to grayscale if necessary,
    gray_image = image if is_grayscale(image) else to_grayscale(image)
    # Detect faces on it.
    faces_rectangles = detect_faces(
        cascade, gray_image, scale_factor, min_neighbours
    )
    # Crop them out and return as a list.
    faces = []
    for face_rectangle in faces_rectangles:
        faces.append(crop(image, face_rectangle))
    return faces


def augment_images(images, apply_flip, apply_noise, noise_intensity):
    """
    Create augmented list of images from list of original images.
    :param images: list of original images
    :param apply_flip: whether to augment with flip or not
    :param apply_noise:  whether to apply noise to the images or not
    :param noise_intensity: intensity of applied noise
    :return: augmented list of images
    """
    new_images = list(images)
    # Apply flip if requested.
    if apply_flip:
        new_images += [flip(image) for image in new_images]
    # Apply noise if requested.
    if apply_noise:
        new_images += [
            add_noise(image, noise_intensity) for image in new_images
        ]
    return new_images


def convert_to_colorspace(images, grayscale, rgb):
    """
    Convert list of input images to the target colorspace
    :param images: list of images to convert
    :param grayscale: whether to convert images to grayscale
    :param rgb: whether to convert images to rgb color space
    :return: converted images
    """
    # Convert to rgb space if requested.
    if rgb:
        new_images = [
            image if is_rgb(image) else to_rgb(image) for image in images
        ]
    # Convert to grayscale otherwise.
    else:
        new_images = [
            image if is_grayscale(image) else to_grayscale(image)
            for image in images
        ]
    return new_images


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
    # Find faces on the image and create list of new images from them.
    faces = extract_faces(cascade, in_image, scale_factor, min_neighbours)
    # Apply data augmentation.
    augmented_images = augment_images(
        faces, apply_flip, apply_noise, noise_intensity
    )
    # Convert obtained images to target format.
    converted_images = convert_to_colorspace(augmented_images, grayscale, rgb)
    # Resize images to desired size and normalize them.
    resized_images = [resize(image, target_size) for image in converted_images]
    normalized_images = [
        image if is_float(image) else normalize(image)
        for image in resized_images
    ]
    # Return both transformed and augmented images.
    return normalized_images, augmented_images
