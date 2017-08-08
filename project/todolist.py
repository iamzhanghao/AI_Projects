from shutil import copyfile
from project.utils import get_data


data = get_data(split="1", size="100X", platform="Windows", user="Zhang Hao")

dst = "C:\\Users\Hao\Desktop\\test2\\"

count = 0
for testdata in data['test']:
    if testdata[1] == [1,0]:
        type_d = "B"
    else:
        type_d = "M"
    copyfile(testdata[0], dst+"testfile_"+str(count)+"_"+type_d+".png")
    count += 1
