import numpy as np
import os
import json

d = {"key": 1, "second key": 3}

print(os.getcwd())

dir = "main"
file = "file.npy"

filepath = os.path.join(dir, file)

with open(filepath, "w") as outfile:
    json.dump(d, outfile)
