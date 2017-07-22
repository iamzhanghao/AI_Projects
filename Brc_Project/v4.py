from Brc_Project.utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

data = get_data(split="2", size="100X", platform="Windows")

# Convolution Layer 1.
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 32  # There are 16 of these filters.

# Convolution Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 32  # There are 36 of these filters.

# Convolution Layer 3.
filter_size3 = 5  # Convolution filters are 5 x 5 pixels.
num_filters3 = 64  # There are 36 of these filters.

# Fully-connected layer.
fc_size1 = 64  # Number of neurons in fully-connected layer.
fc_size2 = 2

# We know that MNIST images are 28 pixels in each dimension.
img_size = 64

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 2

''' Testing Only'''
im_path = data['test'][0][0]
img = read_img(im_path, crop=img_size)
''' Helper Functions'''


def plot_images(img):
    img = Image.fromarray(img)
    img.show()


def new_weights(shape, stddev=0.001):
    return tf.Variable(tf.random_normal(shape, stddev=stddev))


def new_biases(length, stddev=0.001):
    return tf.Variable(tf.random_normal(shape=[length], stddev=stddev))


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(layer_in,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(layer_in, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights, biases


def new_conv_layer(layer_in,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   pooling='max',  # Use 2x2 max-pooling.
                   stddev=0.001,
                   conv_strides=1,
                   pooling_size=3,
                   pooling_strides=2):
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape, stddev=stddev)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters, stddev=stddev)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=layer_in,
                         filter=weights,
                         strides=[1, conv_strides, conv_strides, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if pooling == 'max':
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pooling_size, pooling_size, 1],
                               strides=[1, pooling_strides, pooling_strides, 1],
                               padding='SAME')

    if pooling == 'avg':
        layer = tf.nn.avg_pool(value=layer,
                               ksize=[1, pooling_size, pooling_size, 1],
                               strides=[1, pooling_strides, pooling_strides, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights, biases


''' TensorFlow Graphs'''


def create():
    # Image flat
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    # Actual Image
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    # print(x_image.shape)

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    # print(y_true.shape)
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Convolution Layer 1
    layer_conv1, weights_conv1, biases_conv1 = new_conv_layer(layer_in=x_image,
                                                              num_input_channels=num_channels,
                                                              filter_size=filter_size1,
                                                              num_filters=num_filters1,
                                                              pooling='max',
                                                              stddev=0.0001, )

    # Convolution Layer 2
    layer_conv2, weights_conv2, biases_conv2 = new_conv_layer(layer_in=layer_conv1,
                                                              num_input_channels=num_filters1,
                                                              filter_size=filter_size2,
                                                              num_filters=num_filters2,
                                                              pooling='avg',
                                                              stddev=0.01)

    # Convolution Layer 3
    layer_conv3, weights_conv3, biases_conv3 = new_conv_layer(layer_in=layer_conv2,
                                                              num_input_channels=num_filters2,
                                                              filter_size=filter_size2,
                                                              num_filters=num_filters3,
                                                              pooling='avg',
                                                              stddev=0.0001)

    layer_flat, num_features = flatten_layer(layer_conv3)
    print(layer_flat)

    layer_fc1, weights_fc1, biases_fc1 = new_fc_layer(layer_in=layer_flat,
                                                      num_inputs=num_features,
                                                      num_outputs=fc_size1,
                                                      use_relu=False)

    layer_fc2, weights_fc2, biases_fc2 = new_fc_layer(layer_in=layer_fc1,
                                                      num_inputs=fc_size1,
                                                      num_outputs=fc_size2,
                                                      use_relu=False)
    y_pred = tf.nn.softmax(layer_fc2)

    y_pred_cls = tf.argmax(y_pred, dimension=1)


    pass


create()


