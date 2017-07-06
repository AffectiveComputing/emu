from data import *
from graph import *

SOURCE_DIR = "npy"
LABELS_FILE = "png/emotion.txt"

BATCH_DATA_SIZE = 200
CHANNELS = 3

WIDTH = 128
HEIGHT = 128
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
IN_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

OUT_SIZE = 7
OUT_SHAPE = OUT_SIZE


data_set = DataSet(SOURCE_DIR, LABELS_FILE)

desired_loss = 0.001
current_rate = 0.003

x = tf.placeholder(tf.float32, IN_SHAPE)
prediction, output = get_graph(x, [32, 64], [OUT_SIZE], 5)

correct = tf.placeholder(tf.int64, [BATCH_DATA_SIZE])

loss = tf.reduce_mean(tf.abs(output - tf.one_hot(correct, OUT_SIZE)))
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, correct), dtype=tf.float32))

rate = tf.placeholder(tf.float32, [])
train = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

init_glob = tf.global_variables_initializer()

tf.summary.scalar('loss', loss)
merge = tf.summary.merge_all()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter("log", sess.graph)
    sess.run(init_glob)

    prev_l = l = 100
    i = 1
    j = 1

    while 0 < data_set.size:
        data, labels = data_set.get_data(BATCH_DATA_SIZE, IMG_SHAPE)

        while desired_loss < l:
            _, l, a, p, c, o = sess.run([train, loss, accuracy, prediction, correct, output],
                                        feed_dict={x: data, rate:
                                            current_rate, correct: labels})

            print("Iteration {}\n\tLoss: {}\n\tAccuracy: {}\n\tLearning rate: {}"
                  .format(i, l, a, current_rate))

            if j % 500 == 0:
                current_rate *= 0.99

            prev_l = l
            i += 1
            j += 1

    sess.close()
