import numpy as np
import random
d = np.array([0,0,0,1,1,1,0,0])
zeros = np.where(d == 0)
print(random.choices(zeros, k=4))

f = [1,2,3,4]
random.sample(f,k=2)