import numpy as np
import scipy
import h5py
import tensorflow
from matplotlib import pyplot as plt
import time

start_time = time.time()

def draw_samples(n):
    A1 = np.array([[0.70710678118, 0.70710678118],
                   [-0.70710678118, 0.70710678118]])
    A2 = np.array([[3, 0], [0, 1]])
    A = A1.dot(A2)
    mu1 = np.array([0, 0])
    mu2 = np.array([2.5, 0])
    samples = []

    for i in range(n):
        c = np.random.choice(np.arange(1, 3), p=[0.5, 0.5])
        u = np.random.normal(0, 1, 2)
        if c == 1:
            x = A.dot(u) + mu1
            y = np.random.choice(np.arange(0, 2), p=[0.3, 0.7])
        else:
            x = A.dot(u) + mu2
            y = np.random.choice(np.arange(0, 2), p=[0.6, 0.4])

        samples.append((x, y))
    return samples


def plot_samples(samples):
    for sample in samples:
        if sample[1] == 1:
            color = 'blue'
        else:
            color = 'red'
        plt.scatter(sample[0][0], sample[0][1], color=color)
    plt.show()

samples = draw_samples(1000)

print(time.time()-start_time)

plot_samples(samples)



