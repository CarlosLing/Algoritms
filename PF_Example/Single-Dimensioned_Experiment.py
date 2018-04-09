import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def f(x, k, w):
    return x/2 + 25*x / (1 + x**2) + 8 * np.cos(1.2 * (k-1)) + w


def h(x, v):

    return x**2 / 20 + v


# Divides the sample according to the weights
def divide(X, W):
    Nef = np.round(1/sum(W**2))
    X_s = np.argsort(X)
    index_l = X_s[:Nef]
    index_h = X_s[Nef:]
    return index_l, index_h


L = 5000  # Longitude of the series
# noise vectors:
sigma_w = np.sqrt(2)
sigma_v = 1
w = np.random.normal(loc=0, scale=sigma_w, size=L)
v = np.random.normal(loc=0, scale=sigma_v, size=L)

x_0 = 0.1

# Initialize variables
X_true = np.zeros(L)
Y_true = np.zeros(L)
# Y_measured
Y_estimated = np.zeros(L)
X_estimated = np.zeros(L)

for i in range(L):
    if i == 0:
        X_true[i] = x_0
    else:
        X_true[i] = f(X_true[i], i, w[i])
    Y_true[i] = h(X_true[i], 0)

Y_measured = Y_true + v

N = 100  # Number of points from the particle filter

# Probability and importance functions to weight calculations
q = lambda x, y: np.exp(- (y-h(x, 0))**2 / (2 * sigma_v**2))
pyx = lambda x, y: norm.pdf((y - h(x, 0)) / sigma_v)
pxx = lambda x, x1, k: norm.pdf((x - f(x1, k, 0)) / sigma_w)

# State variables initialization
X_es = np.zeros(N)

# Crossover and mutation parametrers
alpha = 0.4  #
p_m = 0.2  # Mutation parameter
for l in range(L):
    if l == 0:  # initialization
        X_k = np.random.normal(loc=x_0, scale=sigma_w, size=N)
        W_k = np.ones(N)
    else:
        X_k1 = X_k
        W_k1 = W_k

        # Probabilities and importance function initialization
        Q = np.zeros(N)
        PYX = np.zeros(N)
        PXX = np.zeros(N)

        for i in range(N):
            # Sampling
            X_k[i] = np.random.normal(loc = f(X_k1[i], l, 0), scale=sigma_w)

            # Calculate probabilities and importance function
            Q[i] = q(X_k[i], Y_measured[l])
            PYX[i] = pyx(Y_measured[l], X_k[i])
            PYX[i] = pxx(X_k[i], X_k1[i])

        # Compute weights
        Ws = W_k1 * PXX * PYX / Q
        W = Ws / sum(Ws)

        # State variables estimation
        X_es[l] = sum(W * X_k)

        # Perform mutation and crossover operators
        # Crossover
        CL, CH = divide(X_k, W)
        rL = np.random.uniform(low=0, high=1, size=len(CL))
        for x in range(len(CL)):
            arg_H = np.random.choice(CH)
            X_k[CL[x]] = X_k[CL[x]] * alpha + (1-alpha) * X_k[arg_H]
            if rL[x] <= p_m:
                X_k[CL[x]] = 2 * X_k[arg_H] - X_k[CL[x]]



