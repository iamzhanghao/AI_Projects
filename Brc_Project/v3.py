from Brc_Project.utils import *
import tensorflow as tf

data = get_data(split="2", size="100X", platform="Windows")

im_path = data['test'][0][0]
img = read_img(im_path)
print(img.shape)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 460 * 700  # MNIST data input (img shape: 28*28)
n_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, size, strides):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, strides, strides, 1],
                          padding='SAME')


def avgpool2d(x, size, strides):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, strides, strides, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 460, 700, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, size=3, strides=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=1)
    # Max Pooling (down-sampling)
    conv2 = avgpool2d(conv2, size=3, strides=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=1)
    conv3 = avgpool2d(conv3, size=3, strides=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.softmax(fc2, 0)
    fc2 = tf.nn.relu(fc2)

    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal(shape=[5, 5, 1, 32], stddev=0.0001)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal(shape=[5, 5, 32, 32], stddev=0.01)),

    'wc3': tf.Variable(tf.random_normal(shape=[5, 5, 32, 64], stddev=0.0001)),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 64])),
    # 'wd2': tf.Variable(tf.random_normal())
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal(shape=[32], stddev=0.0001)),
    'bc2': tf.Variable(tf.random_normal(shape=[32], stddev=0.01)),
    'bc3': tf.Variable(tf.random_normal(shape=[64], stddev=0.0001)),
    'bd1': tf.Variable(tf.random_normal(shape=[1024])),
    'out': tf.Variable(tf.random_normal(shape=[n_classes]))
}
