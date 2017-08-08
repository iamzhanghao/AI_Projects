from project.haonet import HaoNet
from project.utils import *
import tensorflow as tf



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


def start_new(filename, times, num_of_images):
    train(num_iterations=num_of_images,
          batchsize=128,
          load_weights=None,
          save_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")

    validate(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")
    test(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")
    continuous(filename, times, num_of_images)


def continuous(filename, times, num_of_images):
    for _ in range(times):
        train(num_iterations=200 * num_of_images,
              batchsize=128,
              load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy",
              save_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")
        validate(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")
        test(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\\" + filename + ".npy")



num_of_imgs = 100



print("################################"
      "################################"
      "################################"
      "split1")
data = get_data(split="1", size="200X", platform="Windows", user="Hao")
dataset = Dataset(data, crop=64, num_of_imgs=num_of_imgs)


# start new training
start_new(filename="split1", times=1,num_of_images=num_of_imgs)

# continue training
continuous(filename="split1", times=1,num_of_images=num_of_imgs)

# validate using existing weights
validate(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\set2_100\split1.npy")

# test using existing weights
test(load_weights="C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\set2_100\split1.npy")




