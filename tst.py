import numpy as np

from PIL import Image

img = Image.open("C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png")


# dataset = np.load("C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset2.npy")[()]


def get_image_mean(img):
    r, g, b = 0, 0, 0
    count = 0
    img_np = np.array(img)
    print(img_np.shape)

    for x in range(img_np.shape[0]):
        for y in range(img_np.shape[1]):
            temp = img_np[x][y]

            tempr, tempg, tempb = temp[0], temp[1], temp[2]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    # calculate averages
    return np.array([int((r / count)), int((g / count)), int((b / count))])


mean = get_image_mean(img)
mean = np.array([1,1,1])

tst = np.array([[[0, 1, 2], [4, 5, 6]],
          [[0, 1, 2], [4, 5, 6]],
          [[0, 1, 2], [4, 5, 6]]]
         )

print(tst-mean)
