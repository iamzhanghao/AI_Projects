import numpy as np

from PIL import Image

img = Image.open("C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png")

angle = \
        np.random.choice(
            [0, 90, 180, 270],
            1,
            p=[0.5, 0.2, 0.2, 0.1]
        )

print(img.rotate(angle[0]))