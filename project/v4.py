from project.haonet import HaoNet
from project.utils import *
import tensorflow as tf

dataset = Dataset(path="C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset3.npy")


def train(num_iterations, batchsize, load_weights, save_weights):
    haonet = HaoNet(dataset, params=load_weights)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    haonet.train(session=session, num_iterations=num_iterations, batchsize=batchsize)

    if save_weights is not None:
        haonet.save_params(session, save_weights)


def validate(load_weights):
    haonet = HaoNet(dataset, params=load_weights)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    haonet.validate(session, mode='val')


def test(load_weights):
    haonet = HaoNet(dataset, params=load_weights)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    haonet.validate(session, mode='test')


def start_new():
    train(num_iterations=2 * 1000,
          batchsize=128,
          load_weights="None",
          save_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")

    validate(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")
    test(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")

def continuous(times):
    for _ in range(times):
        train(num_iterations=4 * 1000,
              batchsize=128,
              load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy",
              save_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")

    validate(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")
    test(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params1.npy")


continuous(4)
