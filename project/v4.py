from project.haonet import HaoNet
from project.utils import *
import tensorflow as tf
import copy

''' TensorFlow Graphs'''

# # Image flat
# x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
# # Actual Image
# x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
# # print(x_image.shape)

# Image flat
# Actual Image






dataset = Dataset(path="C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset1.npy")

haonet = HaoNet(dataset, weights=None)

session = tf.Session()
session.run(tf.global_variables_initializer())

haonet.optimize(session=session, num_iterations=200, batchsize=64)

# session.close()

np.save("C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy",session.run(haonet.prams))




