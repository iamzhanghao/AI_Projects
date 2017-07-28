import pprint
import numpy as np
from PIL import Image
from PIL import ImageOps
import random

# from sklearn.utils import shuffle

pp = pprint.PrettyPrinter(indent=5)


def get_data(split="1", size="40X", platform="Windows", user="JunHao"):
    print("Fetching data:\nsplit = " + split + "\nsize = " + size + "\nplatform = " + platform)
    if platform == "Mac":
        data_dir = "/Users/zhanghao/Projects/Project_Dir/"
        project_Dir = "/Users/zhanghao/Projects/AI_Projects/project/breakhissplits_v2/train_val_test_60_12_28/shuffled/split"
    else:
        if user == "JunHao":
            data_dir = "C:\\Users\JunHao\OneDrive\#Term8\\breakhis"
            project_Dir = "C:\\Users\JunHao\OneDrive\#Term8\AIproject\project\\breakhissplits_v2\\train_val_test_60_12_28\shuffled\split"

        else:
            data_dir = "D:\\"
            project_Dir = "C:\\Users\Hao\Projects\AI_Projects\project\\breakhissplits_v2\\train_val_test_60_12_28\shuffled\split"

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
    def __init__(self, data=None, crop=64, path=None, num_of_imgs=10):

        if path is None:

            self.dataset = {
                'train_data': None,
                'train_label': None,
                'val_data': None,
                'val_label': None,
                'test_data': None,
                'test_label': None
            }

            self.init_dataset(data,crop,num_of_imgs)

        else:
            self.dataset = np.load(path)
            self.dataset = self.dataset[()]

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

    def init_dataset(self,data,crop,num_of_imgs):


        data_arr = []
        label_arr = []
        print("Generating dataset...")
        print("Preparing train data...")
        count = 0
        for entry in data['train']:
            count += 1
            if count % 200 == 0:
                print("Progress = ", round(count / len(data['train']) * 100, 2), "%")

            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True)
            for img in imgs:
                data_arr.append(img)
                label_arr.append(entry[1])

        print("Shuffling train data...")
        c = list(zip(data_arr, label_arr))
        random.shuffle(c)
        data_arr, label_arr = zip(*c)
        print("Shuffle done.")

        # data_arr,label_arr = shuffle(data_arr,label_arr)

        self.dataset['train_data'] = np.array(data_arr)
        del data_arr
        self.dataset['train_label'] = np.array(label_arr)
        del label_arr
        print("Train Data: ", self.dataset['train_data'].shape)
        print("Train Label: ", self.dataset['train_label'].shape)

        print("Preparing val data...")
        data_arr = []
        label_arr = []
        for entry in data['val']:
            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True)
            data_arr.append([])
            label_arr.append([])

            for img in imgs:
                data_arr[-1].append(img)
                label_arr[-1].append(entry[1])

        self.dataset['val_data'] = np.array(data_arr)
        self.dataset['val_label'] = np.array(label_arr)
        print("Val Data: ", self.dataset['val_data'].shape)
        print("Val Label: ", self.dataset['val_label'].shape)

        print("Preparing test data...")
        data_arr = []
        label_arr = []
        for entry in data['test']:
            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True)
            data_arr.append([])
            label_arr.append([])
            for img in imgs:
                data_arr[-1].append(img)
                label_arr[-1].append(entry[1])

        self.dataset['test_data'] = np.array(data_arr)
        self.dataset['test_label'] = np.array(label_arr)
        print("Test Data: ", self.dataset['test_data'].shape)
        print("Test Label: ", self.dataset['test_label'].shape)

    def save(self, path):
        print("Saving dataset to "+path)
        np.save(path, self.dataset)

    def next_batch(self, type='train', batch_size=64):
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            batch_x.append(self.dataset[type + '_data'][self.current[type]])
            batch_y.append(self.dataset[type + "_label"][self.current[type]])
            self.current[type] = (self.current[type] + 1) % self.size[type]

        return batch_x, batch_y

        # def get_val(self):
        #     x = []
        #     y = []
        #     for _ in range(self.size['val']):
        #         x.append(self.dataset['val_data'][_])
        #         y.append(self.dataset['val_label'][_])
        #     return x, y


def rotate(img, p0=0.4, p1=0.2, p2=0.2, p3=0.2):
    angle = \
        np.random.choice(
            [0, 90, 180, 270],
            1,
            p=[p0, p1, p2, p3]
        )

    return img.rotate(angle[0])


def mirror(img, p=0.5):
    do = np.random.choice(
        [True, False],
        1,
        p=[p, 1 - p]
    )

    if do[0]:
        return ImageOps.mirror(img)
    else:
        return img


def random_crop(path, patch_size, num_of_imgs, do_rotate=False, do_mirror=False):
    im = Image.open(path)
    size = im.size[0] / 2, im.size[1] / 2
    im.thumbnail(size)

    imgs = []
    for _ in range(num_of_imgs):
        x = random.randint(0, im.size[0] - patch_size)
        y = random.randint(0, im.size[1] - patch_size)
        # print(str(x) + ", " + str(y))
        new_img = im.crop((x, y, x + patch_size, y + patch_size))
        if do_rotate:
            new_img = rotate(new_img)
        if do_mirror:
            new_img = mirror(new_img)

        imgs.append(np.array(new_img))

    return imgs


def prepare():
    # #
    data = get_data(split="2", size="100X", platform="Windows", user="Hao")
    #
    dataset = Dataset(data, crop=64, num_of_imgs=100)
    dataset.save("C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset3.npy")


if __name__ == "__main__":
    prepare()
