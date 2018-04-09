from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt  # Imports dates library

from nupic.encoders import RandomDistributedScalarEncoder
from nupic.encoders.date import DateEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory


from scipy.stats import norm


class Encoder:
    def __init__(self, variable, encoders):
        self.variable = variable
        self.encoders = encoders


def multiencode(encoders, Data, iter):
    res = [0]
    for x in encoders:
        for y in x.encoders:
            if x.variable != '_index':
                exec("enc = " + y + ".encode(Data['" + x.variable + "'][iter])")
            else:
                exec("enc = " + y + ".encode(Data.index[iter])")
            res = np.concatenate([res, enc])
    return res[1:]


# Import data
Data = pd.read_csv('Data/Temperature/ambient_temperature_system_failure.csv',
                   parse_dates=True,
                   index_col='timestamp')

Data = Data.resample('H').bfill().interpolate()
def ma_preprocess(data, window):
    l = len(data)
    Z = np.zeros(l)
    for x in range(l):
        Z[x] = np.mean(data[max(0, x - window // 2):min(l - 1, x + window // 2):])
    MA = pd.DataFrame(Z).set_index(data.index)
    return MA


var_chosen = 'value'
Data = ma_preprocess(Data[var_chosen], 4).rename(columns={0:var_chosen})

Data['Anomaly'] = 0.0
Data['Anomaly_Likelihood'] = 0.0

prec_param = 5
pooler_out = 2024
cell_col = 5

# Value Encoder Resoltion
Res = Data.std()[0]/prec_param
RDSE = RandomDistributedScalarEncoder(resolution=Res)
# We ecndoe now the datas
TODE = DateEncoder(timeOfDay=(21, 1))
WENDE = DateEncoder(weekend=21)

# Spatial Pooler Parameters

var_encoders = {Encoder('value', ['RDSE'])}
# Encoder('_index', ['TODE'])}

encoder_width = 0
for x in var_encoders:
    for y in x.encoders:
        exec("s = " + y + ".getWidth()")
        encoder_width += s


SP = SpatialPooler(inputDimensions=encoder_width,
                   columnDimensions=pooler_out,
                   potentialPct=0.8,
                   globalInhibition=True,
                   numActiveColumnsPerInhArea=pooler_out//50,  # Gets 2% of the total area
                   boostStrength=1.0,
                   wrapAround=False)
TM = TemporalMemory(columnDimensions=(pooler_out,),
                    cellsPerColumn=cell_col)


# Train Spatial Pooler
start = time.time()

active_columns = np.zeros(pooler_out)

print("Spatial pooler learning")

for x in range(len(Data)):
    encoder = multiencode(var_encoders, Data, x)
    # e_val = RDSE.encode(Data['value'][x])
    # e_tod = TODE.encode(Data.index[x])
    # e_wend = WENDE.encode(Data.index[x])
    # encoder = np.concatenate([e_val])
    SP.compute(encoder, True, active_columns)

end = time.time()
print(end - start)


print("Temporal pooler learning")

start = time.time()

A_score = np.zeros(len(Data))
for x in range(len(Data)):
    encoder = multiencode(var_encoders, Data, x)
    # e_val = RDSE.encode(Data['value'][x])
    # e_tod = TODE.encode(Data.index[x])
    # e_wend = WENDE.encode(Data.index[x])
    # encoder = np.concatenate([e_val])
    SP.compute(encoder, False, active_columns)
    col_index = active_columns.nonzero()[0]
    TM.compute(col_index, learn=True)
    if x > 0:
        inter = set(col_index).intersection(Prev_pred_col)
        inter_l = len(inter)
        active_l = len(col_index)
        A_score[x] = 1 - (inter_l / active_l)
        Data.iat[x, -2] = A_score[x]
    Prev_pred_col = list(set(x // cell_col for x in TM.getPredictiveCells()))

end = time.time()
print(end - start)


W = 72
W_prim = 5
eps = 1e-6


AL_score = np.zeros(len(Data))
for x in range(len(Data)):
    if x > 0:
        W_vec = A_score[max(0, x-W): x]
        W_prim_vec = A_score[max(0, x-W_prim): x]
        AL_score[x] = 1 - 2*norm.sf(abs(np.mean(W_vec)-np.mean(W_prim_vec))/max(np.std(W_vec), eps))
        Data.iat[x, -1] = AL_score[x]

dataplot = Data.copy()
dataplot['Anomaly_flag'] = dataplot['Anomaly_Likelihood'] > 0.95
dataplot['Anomaly'] = dataplot['Anomaly_flag'] * dataplot['value']


# Plots

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly'], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("HTM Anomaly detection + Prerocess: Simple")
plt.show()


a = dt.datetime(2014, 1, 20)  # Fixes the start date
b = dt.datetime(2014, 3, 1)  # Fixes the start date

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'][a:b])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly'][a:b], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("HTM Anomaly detection + Prerocess: Detail Simple")
plt.show()

a = dt.datetime(2013, 12, 1)
b = dt.datetime(2014, 1, 15)

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'][a:b])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly'][a:b], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("HTM Anomaly detection + Prerocess: Detail Simple")
plt.show()

plt.show(block=True)
