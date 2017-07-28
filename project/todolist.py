from random import randint
import numpy as np
from PIL import Image
from PIL import ImageOps


# from project.utils import *

# 1. random image cropping
# input: path to image(700*460)
# output: np array with shape [size,size,3]




# random_crop("C:\\Users\JunHao\OneDrive\#Term8\AIproject\project\\breakhis\\40X\SOB_B_A-14-22549AB-40-001.png", 64)

imgs = random_crop("C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png", 64, 4)

print(imgs)
# dataset = Dataset(path="C:\\Users\JunHao\OneDrive\#Term8\AIproject\project\saved_dataset\dataset1.npy")
# img,l=dataset.next_batch(type='train', batch_size=64)
# 2. mirror / rotate


# 3. image mean
#
# 4. momentum
#
# 5. resize image to 350*230
