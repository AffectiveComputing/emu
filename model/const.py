SOURCE_DIR = "data/npy"

META_FILE = "data/logs/best_model/model.meta"
MODEL_FILE = "data/logs/best_model/model"
CHECKPOINTS_FILE = "data/logs/checkpoints/model"
LABELS_FILE = "data/png/labels.txt"
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
WIDTH = 64
HEIGHT = 64
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
IN_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

CLASSES_COUNT = 7
OUT_SHAPE = CLASSES_COUNT