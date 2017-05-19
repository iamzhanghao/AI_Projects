import numpy as np
import scipy
import h5py
import tensorflow
from matplotlib import pyplot as plt
import time


def distance(v1, v2):
    return sum([(x - y) ** 2 for (x, y) in zip(v1, v2)]) ** (0.5)


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


def nnc(x_hat, Dn):
    select = 0
    nearest_distance = float("inf")
    for i in range(len(Dn)):
        d = distance(x_hat, Dn[i][0])
        if d < nearest_distance:
            nearest_distance = d
            select = i
    return Dn[select][1]


def g(x):
    return -(np.dot(np.array([1, 1]),x) - 1.25)


def f(x):
    if x >= 0:
        return 1
    else:
        return 0


Dn = draw_samples(500)
# plot_samples(samples)
Tk = draw_samples(1000)

print("Nearest Neighbor Classifier")
correct = 0
for sample in Tk:
    predicted = nnc(sample[0], Dn)
    if predicted == sample[1]:
        correct += 1
print("Error: " + str(round(1 - correct / 1000., 2)))

print("Second Classifier")
correct = 0
for sample in Tk:
    predicted = f(g(sample[0]))
    if predicted == sample[1]:
        correct += 1
print("Error: " + str(round(1 - correct / 1000., 2)))
