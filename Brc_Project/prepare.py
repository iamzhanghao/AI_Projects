import pprint

pp = pprint.PrettyPrinter(indent=5)


def get_data(split="1", size="40X", platform="Mac"):
    if platform == "Mac":
        data_dir = "/Users/zhanghao/Projects/Project_Dir/"
        project_Dir = "/Users/zhanghao/Projects/AI_Projects/Brc_Project/breakhissplits_v2/train_val_test_60_12_28/shuffled/split"
    else:
        data_dir = "E:\\"
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
                line.replace("/", "\\")
            res = line.split(" ")

            if len(res) == 2:
                if platform != "Mac":
                    res[0].replace("/", "\\")
                data_set[set].append((data_dir + res[0], res[1]))
            line = f.readline()
    return data_set


data = get_data(split="2", size="100X", platform="Windows")
pp.pprint(data)
