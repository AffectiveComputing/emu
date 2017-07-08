"""
This module is responsible for cropping faces out of input images.
"""


import cv2
import argparse
from os import path, listdir


__author__ = ["Michał Górecki"]


def verify_detection(image, faces_rectangles):
    """
    Show rectangles around detected faces and ask user if they are correct.
    :param image: analyzed image
    :param faces_rectangles: detected bounding rectangles for faces
    :return: boolean value indicating if the verification passed or not
    """
    # Constants describing rectangle shown for detection.
    FACE_RECTANGLE_COLOR = (255, 0, 0)
    FACE_RECTANGLE_THICKNESS = 2
    # Copy original image and convert it back to RGB color space to make
    # rectangle more visible.
    image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    for (x, y, w, h) in faces_rectangles:
        cv2.rectangle(
            image_copy, (x, y), (x + w, y + h),
            FACE_RECTANGLE_COLOR, FACE_RECTANGLE_THICKNESS
        )
    # Show image with indicated detections and wait for user input.
    cv2.imshow("Verification", image_copy)
    if chr(cv2.waitKey()) == "y":
        return True
    else:
        return False


def extract_faces(cascade, image, verify, scale_factor=1.05, min_neighbours=5):
    """
    Extract faces from image using given cascade.
    :param cascade: cascade detection object
    :param image: analyzed image
    :param verify: states whether detection result should be verified or not
    :param scale_factor: detection parameter
    :param min_neighbours: detection parameter
    :return: image parts with faces detected in them
    """
    # Constant count of grayscale image dimensions.
    GRAYSCALE_DIMENSION_COUNT = 2
    # If input image is not an grayscale image convert it to grayscale.
    if len(image.shape) > GRAYSCALE_DIMENSION_COUNT:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find faces, optionally verify and extract their image parts.
    faces_rectangles = cascade.detectMultiScale(
        image, scale_factor, min_neighbours
    )
    # Return empty list if verification fails.
    if verify and not verify_detection(image, faces_rectangles):
        return []
    # Else crop out detected faces and return them as a list.
    faces = []
    for (x, y, w, h) in faces_rectangles:
        faces.append(image[y:y + h, x:x + w])
    return faces


def export_faces_from_file(cascade, in_path, out_path, verify):
    """
    Export faces from a single input image file.
    :param in_path: path to the input file
    :param out_path: path to the output directory
    :param verify: boolean used to activate results verification
    :return: -
    """
    image = cv2.imread(in_path)
    # If image was successfully read, extract faces from it and save to the
    # separate files in the output directory.
    if not image is None:
        filename, extension = path.splitext(path.basename(in_path))
        faces = extract_faces(cascade, image, verify)
        for face, i in zip(faces, range(len(faces))):
            cv2.imwrite(
                path.join(out_path, filename + "_{}".format(i) + extension),
                face
            )


def create_parser():
    """
    Create command line parser for this script.
    :return: created parser
    """
    parser = argparse.ArgumentParser(
        description="This script detects and extracts faces visible on input "
                    "image files and stores them in separate files. CAUTION: "
                    "during verification \"y\" key press accepts results."
    )
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
    parser.add_argument(
        "cascade_path", type=str,
        help="Path to the cascade xml file used in detection on faces."
    )
    parser.add_argument(
        "-v", "--verification", action="store_true",
        help="States if extraction results should be displayed for "
             "verification purposes. In this mode pressing \"y\" accepts "
             "detection results and saves extracted faces. Pressing anything "
             "else results in discarding extracted image parts."
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
    cascade = cv2.CascadeClassifier(arguments.cascade_path)
    # If input is a single file path convert only the file pointed by it.
    if path.isfile(arguments.in_path):
        export_faces_from_file(
            cascade, arguments.in_path, arguments.out_path,
            arguments.verification
        )
    # Else convert all files in the given directory path.
    elif path.isdir(arguments.in_path):
        for filename in listdir(arguments.in_path):
            export_faces_from_file(
                cascade, path.join(arguments.in_path, filename),
                arguments.out_path, arguments.verification
            )


if __name__ == "__main__":
    main()
