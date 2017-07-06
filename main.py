from const import *
from data import DataSet
from train import Trainer

trainer = Trainer()
data_set = DataSet(SOURCE_DIR, LABELS_FILE)

trainer.set_logs(True)
trainer.train_graph(data_set, 0.003, 0.001, 100000, 500, 0.99)