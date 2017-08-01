# %matplotlib inline
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import random
from PIL import Image

pic = "C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png"
img = mpimg.imread(pic)
img = img[:, :, :3]
# img = Image.open("C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png")
# img = np.array(img)
w, h, d = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))


def recreate_image(palette, labels, w, h):
    d = palette.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = palette[labels[label_idx]]
            label_idx += 1
    return image


plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (16.8 million colors)')
plt.imshow(img)

# a)
sample = np.zeros(shape=(1000, 3))
for i in range(1000):
    sample[i] = image_array[random.randrange(0, w * h)]

kmeans = KMeans(n_clusters=10, random_state=0).fit(sample)
kmeans_palette = kmeans.cluster_centers_
kmeans_labels = kmeans.predict(image_array)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Compressed image (K-Means)')
im=recreate_image(kmeans_palette, kmeans_labels, w, h)
plt.imshow(im)
print(im.shape)
# random_palette = np.zeros(shape=(32, 3))
# for i in range(32):
#     random_palette[i] = image_array[random.randrange(0, w * h)]
#
# random_labels = pairwise_distances_argmin(random_palette,image_array, axis=0)

# plt.figure(3)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Compressed image (Random)')
# plt.imshow(recreate_image(random_palette, random_labels, w, h))
plt.show()
