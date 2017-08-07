from shutil import copyfile
from project.utils import get_data


data = get_data(split="1", size="40X", platform="Windows", user="Zhang Hao")

dst = ""

count = 0
for testdata in data['test']:
    if testdata[1] == [1,0]:
        type_d = "B"
    else:
        type_d = "M"
    copyfile(testdata[0], "C:\\Users\Hao\Desktop\\test1\\testfile_"+str(count)+"_"+type_d+".png")
    count += 1
