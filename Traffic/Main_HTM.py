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
Data = data_all[a:]  # Subsets the data


lprec_param = {5}
lpooler_out = {2024, 2024*2, 2024*4}
cell_col = 10
lW = {24, 48}
lW_prim = {4}
eps = 1e-6
for prec_param in lprec_param:
    for pooler_out in lpooler_out:
        for W in lW:
            for W_prim in lW_prim:
                Data['Anomaly'] = 0.0
                Data['Anomaly_Likelihood'] = 0.0

                # Vars = {"occupancy_6005"}
                Vars = {"occupancy_6005", "occupancy_t4013", "speed_6005", "speed_7578", "speed_t4013"}
                var_encoders = set()
                # Value Encoder Resoltion
                for x in Vars:
                    exec("RDSE_"+ x +" = RandomDistributedScalarEncoder(resolution=Data['"+ x +"'].std()/prec_param)")
                    var_encoders.add(Encoder(x,["RDSE_" + x]))
                # We encode now the datas
                TODE = DateEncoder(timeOfDay=(21, 1))
                WENDE = DateEncoder(weekend=21)

                # var_encoders.add(Encoder('_index', ['TODE', 'WENDE']))

                encoder_width = 0
                for x in var_encoders:
                    for y in x.encoders:
                        exec("s = " + y + ".getWidth()")
                        encoder_width += s

                SP = SpatialPooler(inputDimensions=encoder_width,
                                   columnDimensions=pooler_out,
                                   potentialPct=0.5,
                                   globalInhibition=True,
                                   numActiveColumnsPerInhArea=pooler_out//50,  # Gets 2% of the total area
                                   boostStrength=1.0,
                                   wrapAround=False)
                TM = TemporalMemory(columnDimensions=(pooler_out,),
                                    cellsPerColumn=cell_col,
                                    predictedSegmentDecrement=0.05)


                # Train Spatial Pooler
                start = time.time()

                active_columns = np.zeros(pooler_out)

                print("Spatial pooler learning")

                for x in range(len(Data)):
                    encoder = multiencode(var_encoders, Data, x)
                    SP.compute(encoder, True, active_columns)

                end = time.time()
                print(end - start)

                print("Temporal pooler learning")

                start = time.time()

                A_score = np.zeros(len(Data))
                for x in range(len(Data)):
                    encoder = multiencode(var_encoders, Data, x)
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

                AL_score = np.zeros(len(Data))
                for x in range(len(Data)):
                    if x > 0:
                        W_vec = A_score[max(0, x-W): max(x-W_prim, 0)]
                        W_prim_vec = A_score[max(0, x-W_prim): x]
                        AL_score[x] = 1 - 2*norm.sf(abs(np.mean(W_vec)-np.mean(W_prim_vec))/max(np.std(W_vec), eps))
                        Data.iat[x, -1] = AL_score[x]

                dataplot = Data.copy()
                dataplot['Anomaly_flag'] = dataplot['Anomaly_Likelihood'] > 0.95



                # Plots

                plt.figure(figsize=(10, 5))
                for x in Vars:
                    plt.plot(dataplot[x])
                for x in list(dataplot[dataplot.Anomaly_flag != 0].index):
                    plt.axvline(x=x, color='black', alpha=0.5)
                plt.xlabel('Time')
                plt.ylabel('Temperature')
                plt.legend()
                plt.title("HTM Anomaly detection: W = " + str(W) +
                          ";Pooler_out = " + str(pooler_out) +
                          ";Precision = " + str(prec_param) +
                          ";W' = " + str(W_prim))
                plt.show()

                plt.figure(figsize=(10, 5))
                plt.plot(dataplot[['Anomaly', 'Anomaly_Likelihood']])
                plt.show()

plt.show(block=True)
