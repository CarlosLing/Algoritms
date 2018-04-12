import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def f(a, b, c):  # Hidden states function
    return a/2 + 25*a / (1 + a**2) + 8 * np.cos(1.2 * b) + c


def h(a, b):  # Measured states function
    return a**2 / 20 + b


# Divides the sample according to the weights
def divide(X, W):
    Nef = int(np.rint(1/sum(W**2)))
    X_s = np.argsort(X)
    index_l = X_s[:Nef]
    index_h = X_s[Nef:]
    print(Nef)
    return index_l, index_h


L = 300  # Longitude of the series
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
        X_true[i] = f(X_true[i-1], i-1, w[i])
    Y_true[i] = h(X_true[i], 0)

Y_measured = Y_true + v

N = 200  # Number of points from the particle filter

# Probability and importance functions to weight calculations


# q = lambda x, y: np.exp(-(y-h(x, 0))**2 / (2 * sigma_v**2))
#  As q is not yet defined we'll try with a unitary importance function
q = lambda x, x1, y: 1
pyx = lambda x, y: norm.pdf((y - h(x, 0)) / sigma_v)
pxx = lambda x, x1, k: norm.pdf((x - f(x1, k, 0)) / sigma_w)

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

        # Initialize Variables vector
        X = np.zeros(N)
        for i in range(N):
            # Sampling
            X[i] = np.random.normal(loc=f(X_k1[i], l, 0), scale=sigma_w)

            # Calculate probabilities and importance function
            PYX[i] = pyx(Y_measured[l], X[i])
            PXX[i] = pxx(X[i], X_k1[i], l)
            Q[i] = q(X[i], X_k1[i], Y_measured[l])  # Still focus on a deeper understanding

        # Compute weights
        Ws = W_k1 * PXX * PYX / Q  # TODO Compute Weights with a function
        W = Ws / sum(Ws)  # Normalize weights to one

        # State variables estimation
        X_estimated[l] = sum(W * X)

        # Perform mutation and crossover operators
        # Crossover
        CL, CH = divide(X, W)
        rL = np.random.uniform(low=0, high=1, size=len(CL))
        for x in range(len(CL)):
            arg_H = np.random.choice(CH)
            X[CL[x]] = X[CL[x]] * alpha + (1-alpha) * X[arg_H]
            if rL[x] <= p_m:  # Mutation
                X[CL[x]] = 2 * X[arg_H] - X[CL[x]]

        # Recalculate weights
        Q = np.zeros(N)
        PYX = np.zeros(N)
        PXX = np.zeros(N)

        for i in range(N):
            # Calculate probabilities and importance function
            PYX[i] = pyx(Y_measured[l], X[i])
            PXX[i] = pxx(X[i], X_k1[i], l)
            Q[i] = q(X[i], X_k1[i], Y_measured[l])  # Still focus on a deeper understanding

        # Compute weights
        Ws = W_k1 * PXX * PYX / Q  # TODO Compute Weights with a function
        W = Ws / sum(Ws)  # Normalize weights to one

        # Resampling
        U = np.sort(abs(np.random.uniform(-1, 0, N)))  # Generate N elements fo Uniform distribution
        sum_w = 0  # Accumulated weights
        i = 0  # weight index

        # Initialize sample values and previous weights associated
        W_k = np.zeros(N)
        X_k = np.zeros(N)

        # Multinomial resampling implementation
        for j in range(N):

            while not sum_w < U[j] <= sum_w + W[i]:
                sum_w += W[i]
                i += 1

            W_k[j] = W[i]
            X_k[j] = X[i]

plt.plot(X_true[:200])
plt.plot(X_estimated[:200])
plt.show()
