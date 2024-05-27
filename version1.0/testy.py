import numpy as np

class clas():
    def __init__(self, b,t):
        self.b = b
        self.t = t
        locs = locals()
        del locs["self"]
        self.locs = locs
    def function_(self):
        print(self.locs)

clas__ = clas(3,4)

clas__.function_()
g = 1

print(g % 3)