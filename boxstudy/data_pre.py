import numpy as np

data = np.load('/home/rl4citygen/InfiniteCityGen/datasets/boxstates/states_patch_max8_relu.npy')
print(data.shape)

print(data[0,2])