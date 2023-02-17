import numpy as np

for i in range(10):
    data = np.load('../data/bat_32_s_64x64x32_{}.npy'.format(i))
    print(data.shape)