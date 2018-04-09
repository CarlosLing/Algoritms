import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


# Read the data
Data = pd.read_csv('Data/Temperature/ambient_temperature_system_failure.csv', parse_dates=True, index_col='timestamp')
one_var = Data['value']

# Select window sizes
W = 500
tiqr = 2.5
tstd = 3


def Anomaly(W, tiqr, tstd):

    Data['Anomaly_val_IQR'] = 0.0
    Data['Anomaly_val_Norm'] = 0.0

    Data['Anomaly_IQR'] = False
    Data['Anomaly_Norm'] = False

    for x in range(len(one_var)):
        if x > 30:
            data = one_var[max(0, x - W):x - 1]
            mean = np.mean(data)
            std = np.std(data)
            q3 = np.percentile(data, 75)
            q1 = np.percentile(data, 25)
            iqr = q3 - q1
            Data.iat[x, -1] = abs(one_var[x] - mean) > std * tstd
            Data.iat[x, -2] = (one_var[x] > q3 + iqr * tiqr) | (one_var[x] < q1 - iqr * tiqr)
            # print("Varianza: ", abs(one_var[x]-median), ", TIQR: ", iqr*tiqr)

    Data['Anomaly_val_IQR'] = Data['value'] * Data['Anomaly_IQR']
    Data['Anomaly_val_Norm'] = Data['value'] * Data['Anomaly_Norm']
    plt.interactive(False)
    plt.figure(figsize=(10,5))
    plt.plot(Data['value'])
    plt.plot(Data[Data.Anomaly_val_Norm != 0]['Anomaly_val_Norm'], 'ro')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.title("Stats Norm Anomaly detection: W = " + str(W) + " t = " + str(tstd))
    plt.show()
    a = dt.datetime(2013, 12, 1)
    b = dt.datetime(2014, 1, 15)
    plot_dfab = Data[a:b]
    plt.figure(figsize=(10, 5))
    plt.plot(plot_dfab['value'])
    plt.plot(plot_dfab[plot_dfab.Anomaly_val_Norm != 0]['Anomaly_val_Norm'], 'ro')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.title("Z-test Norm Anomaly detection: W = " + str(W) + " t = " + str(tstd))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(Data['value'])
    plt.plot(Data[Data.Anomaly_val_IQR != 0]['Anomaly_val_IQR'], 'ro')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.title("Stats IQR Anomaly detection: W = " + str(W) + " t = " + str(tiqr))
    plt.show()


Anomaly(1000, 2, 3)
Anomaly(500, 2, 3)
Anomaly(250, 3, 3)
Anomaly(100, 2, 3)
plt.show(block=True)
