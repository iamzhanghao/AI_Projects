import pprint
import numpy as np
from PIL import Image

pp = pprint.PrettyPrinter(indent=5)


def get_data(split="1", size="40X", platform="Windows"):
    if platform == "Mac":
        data_dir = "/Users/zhanghao/Projects/Project_Dir/"
        project_Dir = "/Users/zhanghao/Projects/AI_Projects/Brc_Project/breakhissplits_v2/train_val_test_60_12_28/shuffled/split"
    else:
        data_dir = "D:\\"
        project_Dir = "C:\\Users\Hao\Projects\AI_Projects\Brc_Project\\breakhissplits_v2\\train_val_test_60_12_28\shuffled\split"

    data_set = {"train": [],
                "val": [],
                "test": []}
    for set in ['train', 'val', 'test']:
        if platform == "Mac":
            f = open(project_Dir + split + "/" + size + "_" + set + ".txt")
        else:
            f = open(project_Dir + split + "\\" + size + "_" + set + ".txt")
        line = f.readline()
        while line != "":
            line = line.replace("\n", "")
            if platform == "Windows":
                line = line.replace("/", "\\")
            res = line.split(" ")

            if len(res) == 2:
                if platform == "Windows":
                    res[0] = res[0].replace("/", '\\')

                if res[1] == '0':
                    data_set[set].append((data_dir + res[0], [1, 0]))
                if res[1] == '1':
                    data_set[set].append((data_dir + res[0], [0, 1]))

            line = f.readline()
    return data_set


def centeredCrop(img, new_height, new_width):
    width = np.size(img, 1)
    height = np.size(img, 0)

    left = np.ceil((width - new_width) / 2.)
    top = np.ceil((height - new_height) / 2.)
    right = np.floor((width + new_width) / 2.)
    bottom = np.floor((height + new_height) / 2.)
    cImg = img.crop((left, top, right, bottom))
    return cImg


def read_img(path, crop=64):
    print("Read File " + path)
    img = Image.open(path)
    img = centeredCrop(img, crop, crop)
    img_arr = np.array(img)
    return img_arr

# #
# data = get_data(split="2", size="100X", platform="Windows")
# pp.pprint(data)
