import tensorflow as tf

from const import OUT_SIZE

weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'traits': tf.Variable(tf.random_normal([32 * 32 * 64, 1024])),
    'classes': tf.Variable(tf.random_normal([1024, OUT_SIZE]))
}

biases = {
    'conv1': tf.Variable(tf.random_normal([32])),
    'conv2': tf.Variable(tf.random_normal([64])),
    'traits': tf.Variable(tf.random_normal([1024])),
    'classes': tf.Variable(tf.random_normal([OUT_SIZE]))
}


def leaky_relu(x):
    return tf.maximum(0.1 * x, x)


def get_conv(x, W, b, strides=1):
    return leaky_relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME'), b))


def get_poll(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_regular(x, W, b):
    return leaky_relu(tf.add(tf.matmul(x, W), b))


def get_net(x):
    conv_1 = get_conv(x, weights["conv1"], biases["conv1"])

    poll_1 = get_poll(conv_1)

    conv_2 = get_conv(poll_1, weights["conv2"], biases["conv2"])

    poll_2 = get_poll(conv_2)

    traits = get_regular(tf.reshape(poll_2, [-1, weights['traits'].get_shape().as_list()[0]]), weights['traits'], biases['traits'])

    classes = get_regular(traits, weights['classes'], biases['classes'])

    return tf.argmax(classes, axis=1), classes
