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


# Generates the isolation forest
def i_forest(X, t, psi):
    """
    :param X: input data
    :param t: number of trees
    :param psi: subsampling size
    :return: Returns an isolation forest
    """

    forest = []
    height_limit = int(np.ceil(np.log2(psi)))
    n = X.shape[0]

    for i in range(t):
        sample = X.iloc[np.random.choice(n, psi, replace=False)]
        forest.append(i_tree(sample, 0, height_limit))

    return forest


# Generates a random isolation tree
def i_tree(x, e, l):
    """
    Generates an isolation tree
    :param x: Input data
    :param e: Current Tree Height
    :param l: Height limit
    :return: Inner Node/ Extreme node
    """

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


def c(size):
    h = np.log(size-1) + np.euler_gamma
    return 2*h - (2*(size-1)/size)

# Computes the leght of the path of the tree provided
def path_length(x, T, e):
    """
    :param x: An instance
    :param T: An isolation Tree
    :param e: The current path lenght
    :return:
    """
    if isinstance(T, ExNode):
        return e + c(T.size) # TODO paper says toadd some qty, revise literature

    attribute = T.s_atr
    if x[attribute] < T.s_val:
        return path_length(x, T.left, e+1)
    else:
        return path_length(x, T.right, e+1)


# Main Program
Data = pd.read_csv('Satellite/Satellite.csv', parse_dates=True)

# Parameters to be set:
samplingFactor = 0.1
t = 20  # Number of trees

# Calculates the sampling size
m = int(Data.shape[0] * samplingFactor)

forest = i_forest(Data, t, m)

