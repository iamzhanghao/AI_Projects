import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.placeholder("float", None)
y = tf.placeholder("float", None)

f_xy = x ** 3 * y ** 2 - 7 * x ** 2 * y + x - 12


def badrun():
    with tf.Session() as session:
        result = session.run(y)
        print(result)


def goodrun():
    feed_dict = {x: [i * 0.0001 for i in range(-1000, 1000)],
                 y: [3 for i in range(-1000, 1000)]}

    with tf.Session() as session:
        # implicit use of a feed dictionary ... what is { ...  }
        result = session.run(f_xy, feed_dict=feed_dict)
        print(result)
        plt.plot(feed_dict[x],result)
        plt.show()


if __name__ == '__main__':
    goodrun()
