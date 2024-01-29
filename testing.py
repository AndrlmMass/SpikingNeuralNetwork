import numpy as np


arr = np.array([[1,1,1,1,1],[0,0,0,0,0], [0,0,0,0,0]])
print(arr)
print(np.where(np.all(arr == 0, axis = 1))[0])
