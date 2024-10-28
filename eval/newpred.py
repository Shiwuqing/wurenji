import numpy as np


zeros_array = np.zeros((2000, 1))

# 将数组保存为pred.npy文件
np.save('pred.npy', zeros_array)
