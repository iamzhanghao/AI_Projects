import pprint
import numpy as np
from PIL import Image

pp = pprint.PrettyPrinter(indent=5)


def get_data(split="1", size="40X", platform="Windows"):
    print("Fetching data:\nsplit = " + split + "\nsize = " + size + "\nplatform = " + platform)
    if platform == "Mac":
        data_dir = "/Users/zhanghao/Projects/Project_Dir/"
        project_Dir = "/Users/zhanghao/Projects/AI_Projects/project/breakhissplits_v2/train_val_test_60_12_28/shuffled/split"
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
    # print("Read File " + path)
    img = Image.open(path)
    img = centeredCrop(img, crop, crop)
    img_arr = np.array(img)
    return img_arr


class Dataset:
    def __init__(self, data=None, crop=64, path=None):

        if path is None:
            self.dataset = {
                'train_data': None,
                'train_label': None,
                'val_data': None,
                'val_label': None,
                'test_data': None,
                'test_label': None
            }

            data_arr = []
            label_arr = []
            print("Generating dataset...")
            for entry in data['train']:
                data_arr.append(read_img(entry[0], crop=crop))
                label_arr.append(entry[1])
            self.dataset['train_data'] = np.array(data_arr)
            self.dataset['train_label'] = np.array(label_arr)
            print("Train Data: ", self.dataset['train_data'].shape)
            print("Train Label: ", self.dataset['train_label'].shape)

            data_arr = []
            label_arr = []
            for entry in data['val']:
                data_arr.append(read_img(entry[0], crop=crop))
                label_arr.append(entry[1])
            self.dataset['val_data'] = np.array(data_arr)
            self.dataset['val_label'] = np.array(label_arr)
            print("Val Data: ", self.dataset['val_data'].shape)
            print("Val Label: ", self.dataset['val_label'].shape)

            data_arr = []
            label_arr = []
            for entry in data['test']:
                data_arr.append(read_img(entry[0], crop=crop))
                label_arr.append(entry[1])
            self.dataset['test_data'] = np.array(data_arr)
            self.dataset['test_label'] = np.array(label_arr)
            print("Test Data: ", self.dataset['test_data'].shape)
            print("Test Label: ", self.dataset['test_label'].shape)

        else:
            self.dataset = np.load(path)
            self.dataset = self.dataset[()]
            print(type(self.dataset))
            print("Train Data: ", self.dataset['train_data'].shape)
            print("Train Label: ", self.dataset['train_label'].shape)
            print("Val Data: ", self.dataset['val_data'].shape)
            print("Val Label: ", self.dataset['val_label'].shape)
            print("Test Data: ", self.dataset['test_data'].shape)
            print("Test Label: ", self.dataset['test_label'].shape)

        self.current = {
            'train': 0,
            'val': 0,
            'test': 0
        }

        self.size = {
            'train': self.dataset['train_data'].shape[0],
            'val': self.dataset['val_data'].shape[0],
            'test': self.dataset['test_data'].shape[0]
        }

    def save(self, path):
        np.save(path, self.dataset)

    def next_batch(self, type='train', batch_size=64):
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            batch_x.append(self.dataset[type + '_data'][self.current[type]])
            batch_y.append(self.dataset[type + "_label"][self.current[type]])
            self.current[type] = (self.current[type] + 1) % self.size[type]

        return batch_x, batch_y


# #
# data = get_data(split="2", size="100X", platform="Windows")
#
# dataset = Dataset(data)
# dataset.save("C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset1.npy")

# dataset = Dataset(path="C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset1.npy")


