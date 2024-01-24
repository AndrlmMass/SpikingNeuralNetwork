import numpy as np
sign_array = np.array([[-1,1,0,1,1,-1],[0,-1,-1,0,1,1], [1,0,1,1,0,1]])

d = np.mean(np.all(sign_array[:,:] >= 0, axis=1))
print(d)