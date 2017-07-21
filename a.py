import numpy as np
import pprint

pp = pprint.PrettyPrinter()




path = 'C:\\Users\Hao\Projects\AI_Projects\week5\crop\largefiles\\bvlc_alexnet.npy'

a = np.load(path,encoding = 'bytes').item()
pp.pprint(a)