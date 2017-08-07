import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from project.utils import *


class HaoNet:
    def __init__(self, dataset, params=None):
        self.init_variables()
        self.dataset = dataset
        if params is None:
            self.params = {
                'weights': {
                    'conv1': None,
                    'conv2': None,
                    'conv3': None,
                    'fc1': None,
                    'fc2': None
                },
                'biases': {
                    'conv1': None,
                    'conv2': None,
                    'conv3': None,
                    'fc1': None,
                    'fc2': None
                }

            }
        else:
            params = np.load(params)
            params = params[()]
            self.params = {
                'weights': {
                    'conv1': tf.Variable(params['weights']['conv1']),
                    'conv2': tf.Variable(params['weights']['conv2']),
                    'conv3': tf.Variable(params['weights']['conv3']),
                    'fc1': tf.Variable(params['weights']['fc1']),
                    'fc2': tf.Variable(params['weights']['fc2'])
                },
                'biases': {
                    'conv1': tf.Variable(params['biases']['conv1']),
                    'conv2': tf.Variable(params['biases']['conv2']),
                    'conv3': tf.Variable(params['biases']['conv3']),
                    'fc1': tf.Variable(params['biases']['fc1']),
                    'fc2': tf.Variable(params['biases']['fc2'])
                }
            }

        self.create()

    def init_variables(self):
        # Convolution Layer 1.
        self.filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
        self.num_filters1 = 32  # There are 16 of these filters.

        # Convolution Layer 2.
        self.filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
        self.num_filters2 = 32  # There are 36 of these filters.

        # Convolution Layer 3.
        self.filter_size3 = 5  # Convolution filters are 5 x 5 pixels.
        self.num_filters3 = 64  # There are 36 of these filters.

        # Fully-connected layer.
        self.fc_size1 = 64  # Number of neurons in fully-connected layer.
        self.fc_size2 = 2

        # We know that MNIST images are 28 pixels in each dimension.
        self.img_size = 64

        # Images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_size * self.img_size

        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)

        # Number of colour channels for the images: 1 channel for gray-scale.
        self.num_channels = 3

        # Number of classes, one class for each of 10 digits.
        self.num_classes = 2

        self.KEEPPROB = 0.7

    def plot_images(self, img):
        img = Image.fromarray(img)
        img.show()

    def new_weights(self, shape, stddev=0.001):
        return tf.Variable(tf.random_normal(shape, stddev=stddev))

    def new_biases(self, length, stddev=0.001):
        return tf.Variable(tf.random_normal(shape=[length], stddev=stddev))

    def flatten_layer(self, layer):
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

    def new_fc_layer(self, layer_in,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True,
                     weights=None,
                     biases=None):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.


        if weights is None or biases is None:
            weights = self.new_weights(shape=[num_inputs, num_outputs])
            biases = self.new_biases(length=num_outputs)
        else:
            weights = weights
            biases = biases

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(layer_in, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        layer = tf.nn.dropout(layer,self.KEEPPROB)

        return layer, weights, biases

    def new_conv_layer(self, layer_in,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       pooling='max',  # Use 2x2 max-pooling.
                       stddev=0.001,
                       conv_strides=1,
                       pooling_size=3,
                       pooling_strides=2,
                       weights=None,
                       biases=None):
        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        if weights is None or biases is None:
            weights = self.new_weights(shape=shape, stddev=stddev)
            biases = self.new_biases(length=num_filters, stddev=stddev)
        else:
            weights = weights
            biases = biases

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
        layer = tf.nn.dropout(layer,self.KEEPPROB)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights, biases

    def create(self):
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels],
                                      name='x_image')
        # print(x_image.shape)

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        # print(y_true.shape)
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        # Convolution Layer 1
        self.layer_conv1, \
        self.params['weights']['conv1'], \
        self.params['biases']['conv1'] = self.new_conv_layer(layer_in=self.x_image,
                                                             num_input_channels=self.num_channels,
                                                             filter_size=self.filter_size1,
                                                             num_filters=self.num_filters1,
                                                             pooling='max',
                                                             stddev=0.0001,
                                                             weights=self.params['weights']['conv1'],
                                                             biases=self.params['biases']['conv1'])

        # Convolution Layer 2
        self.layer_conv2, \
        self.params['weights']['conv2'], \
        self.params['biases']['conv2'] = self.new_conv_layer(layer_in=self.layer_conv1,
                                                             num_input_channels=self.num_filters1,
                                                             filter_size=self.filter_size2,
                                                             num_filters=self.num_filters2,
                                                             pooling='avg',
                                                             stddev=0.01,
                                                             weights=self.params['weights']['conv2'],
                                                             biases=self.params['biases']['conv2'])

        # Convolution Layer 3
        self.layer_conv3, \
        self.params['weights']['conv3'], \
        self.params['biases']['conv3'] = self.new_conv_layer(layer_in=self.layer_conv2,
                                                             num_input_channels=self.num_filters2,
                                                             filter_size=self.filter_size2,
                                                             num_filters=self.num_filters3,
                                                             pooling='avg',
                                                             stddev=0.0001,
                                                             weights=self.params['weights']['conv3'],
                                                             biases=self.params['biases']['conv3'])

        self.layer_flat, self.num_features = self.flatten_layer(self.layer_conv3)

        self.layer_fc1, \
        self.params['weights']['fc1'], \
        self.params['biases']['fc1'] = self.new_fc_layer(layer_in=self.layer_flat,
                                                         num_inputs=self.num_features,
                                                         num_outputs=self.fc_size1,
                                                         use_relu=False,
                                                         weights=self.params['weights']['fc1'],
                                                         biases=self.params['biases']['fc1'])

        self.layer_fc2, \
        self.params['weights']['fc2'], \
        self.params['biases']['fc2'] = self.new_fc_layer(layer_in=self.layer_fc1,
                                                         num_inputs=self.fc_size1,
                                                         num_outputs=self.fc_size2,
                                                         use_relu=False,
                                                         weights=self.params['weights']['fc2'],
                                                         biases=self.params['biases']['fc2'])


        self.y_pred = tf.nn.softmax(self.layer_fc2)

        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layer_fc2,
                                                                     labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, session, num_iterations, batchsize):
        # Ensure we update the global variable rather than a local copy.
        total_iterations = 0

        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(total_iterations,
                       total_iterations + num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = self.dataset.next_batch(type='train', batch_size=batchsize)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x_image: x_batch,
                               self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            session.run(self.optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if i % 100 == 0:
                # Calculate the accuracy on the training-set.
                acc = session.run(self.accuracy, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

                # Print it.
                print(msg.format(i + 1, acc))

        # Update the total number of iterations performed.
        total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def validate(self, session, mode):
        correct = 0
        total = len(self.dataset.dataset[mode + '_data'])

        for i in range(total):
            x = self.dataset.dataset[mode + '_data'][i]
            y = self.dataset.dataset[mode + '_label'][i]

            feed_dict_train = {self.x_image: x,
                               self.y_true: y}

            y_pre_cls = session.run(self.y_pred_cls, feed_dict=feed_dict_train)

            y_true_cls = self.dataset.dataset[mode + '_label'][i][0][1]
            f = float(sum(y_pre_cls)) / len(y_pre_cls)

            if f > 0.5:
                y_pre_cls = 1
            else:
                y_pre_cls = 0
            if y_pre_cls == y_true_cls:
                correct += 1
        print(mode + " accuracy = " + str(round(float(correct) / total * 100, 2)) + "%")

    def classify(self, list_of_imgs_path):

        results = []
        testset = []
        for img_path in list_of_imgs_path:
            if check_size(img_path,64):
                imgs = random_crop(path=img_path, patch_size=64, num_of_imgs=100, do_rotate=True, do_mirror=True,
                                   sub_mean=True)
                testset.append(np.array(imgs))
            else:
                testset.append("Error")
        print(testset)


        testset = np.array(testset)
        print(testset.shape)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for i in range(testset.shape[0]):
            if testset[i] != "Error":
                x = testset[i]
                y = []
                for j in range(testset.shape[1]):
                    y.append([0, 0])
                y = np.array(y)

                feed_dict = {self.x_image: x,
                             self.y_true: y}

                y_pre_cls = session.run(self.y_pred_cls, feed_dict=feed_dict)
                f = float(sum(y_pre_cls)) / len(y_pre_cls)

                if f > 0.5:
                    y_pre_cls = 1
                else:
                    y_pre_cls = 0
                results.append(y_pre_cls)
                session.close()
            else:
                results.append("Error")
            print(results)

        return results


    def save_params(self, session, path):
        np.save(path, session.run(self.params))
