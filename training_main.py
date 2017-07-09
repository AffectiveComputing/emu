
from model_building.const import SOURCE_DIR, LABELS_FILE
from model_building.data import DataSet
from model_building.model import Model

if __name__ == "__main__":
    model = Model()
    model.logs = True
    data_set = DataSet(SOURCE_DIR, LABELS_FILE)

    model.train(data_set, 0.003, 0.001, 100000, 5, 0.99, 100, 5, 20)