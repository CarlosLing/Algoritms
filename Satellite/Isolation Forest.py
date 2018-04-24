import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf


# Nodes classes definition
# Internal node
class InNode:
    def __init__(self, left, right, att, val):
        self.left = left
        self.right = right
        self.s_atr = att
        self.s_val = val


# External node
class ExNode:
    def __init__(self, size):
        self.size = size

# Generates a random isolation tree
def i_tree(x, e, l):

    if e >= l or x.size <= 1:
        return ExNode(x.shape[0])

    else:
        q = random.choice(x.columns)
        [v_max, v_min] = [max(x[q]), min(x[q])]
        p = np.random.uniform(v_min, v_max)
        # TODO: Consider p as a simple random choice of a value of q
        # Filtering
        xl = x[x[q] < p]
        xr = x[x[q] >= p]
        return InNode(i_tree(xl, e+1, l), i_tree(xr, e+1, l), q, p)


Data = pd.read_csv('Satellite.csv')

start = time.time()
a = i_tree(Data, 1, 10)
end = time.time()

print(end-start)
