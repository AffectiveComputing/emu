SOURCE_DIR = "data/npy"

META_FILE = "data/models/model_building.meta"
MODEL_FILE = "data/models/model_building"
CHECKPOINTS_FILE = "data/checkpoints/models"
LABELS_FILE = "data/png/emotion.txt"

CHANNELS = 3

WIDTH = 128
HEIGHT = 128
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
IN_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

CLASSES_NUM = 7
OUT_SHAPE = CLASSES_NUM