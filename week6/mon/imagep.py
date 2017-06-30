import numpy as np
from PIL import Image


imname = 'C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\img0.png'

image = Image.open(imname)
image = np.array(image)
print(type(image))
print(image)
print(image.shape)
img = Image.fromarray(image)
img.save('C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\modified_img0.png')
img.show()