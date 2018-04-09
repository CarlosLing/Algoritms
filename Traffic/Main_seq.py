import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import dtw
import sklearn.preprocessing as prepro
import random
import time

# Loading Data
s = set()
s.add('occupancy_6005')
s.add('occupancy_t4013')
s.add('speed_6005')
s.add('speed_7578')
s.add('speed_t4013')
s.add('TravelTime_387')
s.add('TravelTime_451')

for x in s:
    exec(x + " = pd.read_csv(filepath_or_buffer='Data/Traffic/" + x + ".csv', parse_dates=True, index_col='timestamp')")
    exec(x + " = " + x + ".rename(columns={'value': x})")
    exec(x + " = " + x + ".resample('H').mean()")


# Concatenates the variables we are interested in
data_all = pd.concat([occupancy_6005, occupancy_t4013, speed_6005, speed_7578, speed_t4013], axis=1)
data_all = data_all.resample('H').bfill().interpolate()

# Subsets only the variables from the date when the 3 speed sensor is active
a = dt.datetime(2015, 9, 8, 12)  # Fixes the start date
DF = data_all[a:]  # Subsets the data

seq_len = 12
res = np.zeros([len(DF)-seq_len, seq_len])
var = 'occupancy_6005'

for x in range(len(DF)):
    if x >= seq_len:
        res[x-seq_len] = DF[var][x - seq_len: x]


# res = prepro.scale(save_len, axis=1)  # Scales all the time series to mean 0 and std 1
n = len(res)  # The number of time series available

# Distance defined to compute the differences between
def d(x, y):
    return (x-y)**2 # we are using euclidean distance


# Computes the whole distances matrix
"""
D_mat = np.zeros([n, n])
start = time.time()
for i in range(n):
    for j in range(n):
        if j > i :
            D_mat[i,j], cm, acm, wp = dtw.dtw(res[i], res[j], d)
        else:
            D_mat[i, j] = D_mat[j, i]
end = time.time()
print(end-start)
"""

# K-medoids algorithm implementation:
# Parameters definition
c = 6  # Number of clusters
m = 1.3  # Fuzzifier parameter

medoids = np.asarray(random.sample(list(range(n)), c))
initial_medoids = np.zeros(c)
# Still have to determine whether there is easier to compute once all the distances and use them as a reference
# or compute them each time
# The execution time in each case might depend on the number of data samples and clusters

# Create membership matrix
U = np.zeros([c, n])
eps = 1e-8  # To avoid zero divisions
iter = 0
print("Iteration: {}; Medoids: {}".format(iter, medoids))

D_mat = np.zeros([n, n])
CD_mat = np.zeros([n, n])
while sum(initial_medoids != medoids) > 0:
    initial_medoids = medoids

    #Computes the memberships
    R_mat = np.zeros([c, n])
    for i in range(n):
        for j in range(c):

            if CD_mat[i, initial_medoids[j]] == 0:
                D_mat[initial_medoids[j], i], cm, acm, wp = dtw.dtw(res[i], res[initial_medoids[j]], d)
                D_mat[i, initial_medoids[j]] = D_mat[initial_medoids[j], i]
                CD_mat[initial_medoids[j], i] = 1
                CD_mat[i, initial_medoids[j]] = 1

            R_mat[j, i] = (1 / max(D_mat[initial_medoids[j], i], eps)) ** (1 / (m - 1))

    R_sum = np.sum(R_mat, axis=0)
    for i in range(n):
        for j in range(c):
            U[j, i] = R_mat[j, i] / R_sum[i]
    cent = np.dot(U ** m, D_mat)
    medoids = np.argmin(cent, axis=1)
    J = np.sum(np.multiply(D_mat[initial_medoids, :], U ** m))
    iter += 1
    print("Iteration: {}; Medoids: {}; Cost Function {}".format(iter, medoids, J))



# Computes the Medoids using the distance Matrix
"""
while sum(initial_medoids != medoids) > 0:
    initial_medoids = medoids
    # Compute the memberships
    R_mat = np.zeros([c, n])
    for i in range(n):  # Computes the distances, or acesses the distances
        for j in range(c):
            R_mat[j, i] = (1/max(D_mat[initial_medoids[j], i], eps)) ** (1/(m-1))
    R_sum = np.sum(R_mat, axis=0)
    for i in range(n):
        for j in range(c):
            U[j, i] = R_mat[j, i] / R_sum[i]
    cent = np.dot(U**m, D_mat)
    medoids = np.argmin(cent, axis=1)
    J = np.sum(np.multiply(D_mat[initial_medoids, :], U**m))
    iter += 1
    print("Iteration: {}; Medoids: {}; Cost Function {}".format(iter, medoids, J))
"""


# Plots the series and also the medoids centers
plt.figure()
DF[var].plot()
for x in medoids:
    DF[var][x:x+seq_len].plot()
