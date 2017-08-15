"""
This module contains image transformations, conversions and checks. It also
contains face detection and extraction routines.
"""

import cv2

__author__ = ["Paweł Kopeć", "Michał Górecki"]

CASCADE_FILE = "data/cascades/haarcascade_frontalface_default.xml"


def crop(image, rectangle):
    """
    Crop out input image's fragment specified by the rectangle.
    :param image: input image
    :param rectangle: rectangle, which indicates cropped area
    :return: cropped original image fragment
    """
    x, y, w, h = rectangle
    return image[y:y + h, x:x + w]


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


def detect_faces(image, cascade, scale_factor, min_neighbours):
    """
    Detect faces visible on the image.
    :param cascade: cascade object used for detection
    :param image: analyzed image
    :param scale_factor: subsequent detections scaling coefficient
    :param min_neighbours: minimum detection neighbours
    :return: detected face rectangles
    """
    return cascade.detectMultiScale(image, scale_factor, min_neighbours)


def extract_faces(image, cascade=load_cascade(CASCADE_FILE), scale_factor=1.05,
                  min_neighbours=5):
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
        gray_image, cascade, scale_factor, min_neighbours
    )
    # Crop them out and return as a list.
    faces = []
    for face_rectangle in faces_rectangles:
        faces.append(crop(image, face_rectangle))
    return faces


def convert_to_colorspace(images, color_space="rgb"):
    """
    Convert list of input images to the target colorspace
    :param images: list of images to convert
    :param color_space: target colorspace
    :return: converted images
    """
    # Convert to rgb space if requested.
    if color_space == "rgb":
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
