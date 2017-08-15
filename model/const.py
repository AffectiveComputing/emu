"""
This module contains all necessary constants needed for building and training
the model.
"""

DATA_SET_DIR = "data/dataset/fer2013.csv"
META_FILE = "data/logs/best_model/model.meta"
MODEL_FILE = "data/logs/best_model/model"
CHECKPOINTS_FILE = "data/logs/checkpoints/model"
TRAIN_LOG_DIR = "data/logs/train"
VALIDATION_LOG_DIR = "data/logs/train"

# Directories that must exist before the training of a model.
DIRS_TO_ENSURE = [
    "data/logs/best_model/",
    "data/logs/checkpoints/model/",
    "data/logs/train/"
]

CHANNELS = 1

# Image attributes.
WIDTH = 48
HEIGHT = 48
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
IN_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

CLASSES_COUNT = 7

