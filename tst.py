from PIL import Image

im = Image.open("C:\\Users\Hao\Desktop\\test1\\large.png").convert("RGB")
print(im.size)