from const import *
from data import DataSet
from model import Model

model = Model()
data_set = DataSet(SOURCE_DIR, LABELS_FILE)

model.set_logs(True)
model.train(data_set, 0.003, 0.001, 100000, 5, 0.99, 200)