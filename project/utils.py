import pprint
import numpy as np
from PIL import Image
from PIL import ImageOps
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

            self.init_dataset(data, crop, num_of_imgs)

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

    def init_dataset(self, data, crop, num_of_imgs):

        data_arr = []
        label_arr = []
        print("Generating dataset...")
        print("Preparing train data...")
        count = 0
        for entry in data['train']:
            count += 1
            if count % 200 == 0:
                print("Progress = ", round(count / len(data['train']) * 100, 2), "%")

            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True,
                               sub_mean=True)
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
            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True,
                               sub_mean=True)
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
            imgs = random_crop(entry[0], patch_size=crop, num_of_imgs=num_of_imgs, do_rotate=True, do_mirror=True,
                               sub_mean=True)
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
        print("Saving dataset to " + path)
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


def get_image_mean(img):
    r, g, b = 0, 0, 0
    count = 0
    img_np = np.array(img)
    # print(img_np.shape)

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


def random_crop(path, patch_size, num_of_imgs, do_rotate=False, do_mirror=False, sub_mean=False):
    im = Image.open(path)
    size = im.size[0] / 2, im.size[1] / 2
    im.thumbnail(size)
    mean = get_image_mean(im)

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
        if sub_mean:
            new_img = new_img - mean

        imgs.append(np.array(new_img))

    return imgs


def prepare():
    # #
    data = get_data(split="1", size="100X", platform="Windows", user="Hao")
    #
    dataset = Dataset(data, crop=64, num_of_imgs=10)
    dataset.save("C:\\Users\Hao\Projects\AI_Projects\project\saved_dataset\dataset6.npy")


def print_path():
    data = get_data(split="1", size="100X", platform="Windows", user="Hao")
    for i in data['test']:
        a = i[0].replace("\\", "\\\\")
        print("\"", end="")
        print(a, end="\",\n")

def recreate_image(palette, labels, w, h):
    d = palette.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = palette[labels[label_idx]]
            label_idx += 1
    return image

def clusterFuck(path):
    img = mpimg.imread(path)
    img = img[:, :, :3]

    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))
    sample = np.zeros(shape=(1000, 3))
    for i in range(1000):
        sample[i] = image_array[random.randrange(0, w * h)]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(sample)
    kmeans_palette = kmeans.cluster_centers_
    kmeans_labels = kmeans.predict(image_array)

    plt.figure(2)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Compressed image (K-Means)')
    im = recreate_image(kmeans_palette, kmeans_labels, w, h)
    # im=Image.fromarray(im.astype('uint8'), 'RGB')
    # im.show()
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    prepare()
    # print_path()
    # pic = "C:\\Users\Hao\Desktop\SOB_B_A-14-22549AB-40-009.png"
    # clusterFuck(pic)
