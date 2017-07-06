from const import *
from data import DataSet
from model import Model
from matplotlib import pyplot as plt

model = Model()
data_set = DataSet(SOURCE_DIR, LABELS_FILE)

#model.train(data_set, 0.003, 0.001, 100000, 5, 0.99, 400)
x = data_set.get_data(1, IMG_SHAPE)
print(x[0].shape)
result = model.infer(x[0])
plt.imshow(x[0][0])
plt.show()
plt.bar(range(7), result)
plt.show()
