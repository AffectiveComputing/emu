from const import *
from data import *
from graph import *

data, labels = get_data(SOURCE_DIR, LABELS_FILE, BATCH_DATA_SIZE, IMG_SHAPE)

desired_loss = 0.001
current_rate = 0.001

x = tf.placeholder(tf.float32, IN_SHAPE)
prediction, output = get_graph(x, [16, 32, 64], [32, 64, OUT_SIZE], 5)

correct = tf.constant(labels, dtype=tf.int64)

loss = tf.reduce_mean(tf.abs(output - tf.one_hot(correct, OUT_SIZE)))
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, correct), dtype=tf.float32))

init_glob = tf.global_variables_initializer()
init_loc = tf.local_variables_initializer()

rate = tf.placeholder(tf.float32, [])
train = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss)

tf.summary.scalar('loss', loss)
merge = tf.summary.merge_all()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter("log", sess.graph)
    sess.run(init_glob)
    sess.run(init_loc)

    prev_l = l = 100
    i = 1
    j = 1

    while desired_loss < l:
        _, l, a, p, c, o = sess.run([train, loss, accuracy, prediction, correct, output],
                                    feed_dict={x: data, rate: current_rate})

        print(i, ". Loss: ", l, ", accuracy: ", a, ", rate: ", current_rate)

        if j % 500 == 0:
            current_rate *= 0.99

        prev_l = l
        i += 1
        j += 1

    sess.close()
