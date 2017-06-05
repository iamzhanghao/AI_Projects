import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = tf.placeholder("float", None)
y = tf.placeholder("float", None)

f_xy = x ** 3 * y ** 2 - 7 * x ** 2 * y + x - 12


def badrun():
    with tf.Session() as session:
        result = session.run(y)
        print(result)


def goodrun():
    raw_dict = {x: [i * 0.0001 for i in range(-1000, 1000)],
                y: [i * 0.0001 for i in range(-1000, 1000)]}

    feed_dict = {x: [],y:[]}
    for i in raw_dict[x]:
        for j in raw_dict[y]:
            feed_dict[x].append(i)
            feed_dict[y].append(j)

    print(len(feed_dict[x]))


    with tf.Session() as session:
        # implicit use of a feed dictionary ... what is { ...  }
        result = session.run(f_xy, feed_dict=feed_dict)
        print(result)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feed_dict[x], feed_dict[y], result)
        plt.show()


if __name__ == '__main__':
    goodrun()
