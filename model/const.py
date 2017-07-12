SOURCE_DIR = "data/npy"

META_FILE = "data/logs/best_model/model.meta"
MODEL_FILE = "data/logs/best_model/model"
CHECKPOINTS_FILE = "data/logs/checkpoints/model"
LABELS_FILE = "data/png/emotion.txt"

CHANNELS = 1

WIDTH = 64
HEIGHT = 64
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
IN_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

CLASSES_COUNT = 7
OUT_SHAPE = CLASSES_COUNT