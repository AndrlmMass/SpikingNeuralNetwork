import numpy as np
num_neurons = 3
weight_array = ([0,0,0],[1,0.5,0],[0.5,0.2,0])
excit_inhib_ratio = 0.8
if np.any(np.all(weight_array == 0, axis=1)):
    print("this was true")
    for j in range(num_neurons):
        if np.all(weight_array[:,j] == 0):
            idx = np.random.choice(num_neurons)
            if idx == j:
                idx = np.random.choice(num_neurons)
            else:
                const = 1 if np.random.rand() < excit_inhib_ratio else -1
                weight_array[idx, j] = np.random.rand() * const
        if np.any(np.all(weight_array == 0, axis=0)) == False:
            break
print(weight_array)