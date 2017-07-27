from random import randint
import numpy as np
from PIL import Image
from PIL import ImageOps
# from project.utils import *

# 1. random image cropping
# input: path to image(700*460)
# output: np array with shape [size,size,3]

def rotate(img):
    i=randint(0,3)
    img.rotate(90*i).show()

def random_crop(path, patch_size):
    im=Image.open(path)
    x=randint(0, 700-patch_size)
    y=randint(0, 460-patch_size)
    print(str(x)+", "+str(y))
    imnew=im.crop((x,y,x+patch_size,y+patch_size))
    imnew.show()
    #mirror
    ImageOps.mirror(imnew).show()
    #randomly rotate the image
    rotate(imnew)
    print(np.array(imnew).shape)
    #resized
    size=175, 115
    im.thumbnail(size)
    im.show()

random_crop("C:\\Users\JunHao\OneDrive\#Term8\\breakhis\\40X\SOB_B_A-14-22549AB-40-001.png",64)
# dataset = Dataset(path="C:\\Users\JunHao\OneDrive\#Term8\AIproject\project\saved_dataset\dataset1.npy")
# img,l=dataset.next_batch(type='train', batch_size=64)
# 2. mirror / rotate


# 3. image mean
#
# 4. momentum
#
# 5. resize image to 350*230