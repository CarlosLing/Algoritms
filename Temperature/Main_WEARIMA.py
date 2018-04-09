import itertools
import warnings
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime as dt

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

warnings.filterwarnings("ignore")  # specify to ignore warning messages

def pq_arima(data, range_pq):

    # Define the p, q possible parameter combinations
    p = q = range(0, range_pq)
    pq = list(itertools.product(p, q))

    #   warnings.filterwarnings("ignore") # specify to ignore warning messages
    # Chooses the adequated parameters attending to the AIC

    aic = -1
    c_param = (0, 0)
    for param in pq:
        try:
            mod = ARMA(data,
                       order=param)
            res = mod.fit(disp=False)
            if aic == -1 or aic > res.aic:
                # print('ARMA{} - AIC:{} - BIC:{}'.format(param, res.aic, res.bic))
                aic = res.aic
                c_param = param
        except:
            continue

    return c_param


DF = pd.read_csv("Data/Temperature/ambient_temperature_system_failure.csv",
                 parse_dates=True,
                 index_col='timestamp')

Wind = 100  # Window to ARIMA fitting
ErrThred = 0.2  # We are not working with prediction confidence intervals but with relative errors
PredStep = 6
range_pq = 3
Conf_int = 0.05

sample = DF['value'][0:100]

pred_mat = np.zeros((PredStep, PredStep))
conf_int_mat = np.zeros((PredStep, PredStep,2))
weights = np.exp(np.linspace(start=PredStep, stop=1, num=PredStep))

DF['pred'] = 0.0
DF['ci_sup'] = 0.0
DF['ci_inf'] = 0.0

start = time.time()

for x in range(len(DF)):
    if x > 50:
        data_fit = DF['value'][max(0, x-Wind): x]
        X = data_fit.copy()
        res = adfuller(X)
        is_stationary = res[1] < 0.01  # Checks stationarity for the first time
        d = 0
        # Differenciates until stationarity
        while not is_stationary and d < 2:
            X = X.diff().dropna()
            res = adfuller(X)
            is_stationary = res[1] < 0.01
            d += 1
        # Checks the optimal p q parameters
        [p, q] = pq_arima(X, range_pq)

        # Defines and fits the ARIMA model
        mod = ARIMA(data_fit,
                    order=(p, d, q))
        fitted = mod.fit(disp=False)

        # Forecasts the number of steps required
        forecast = fitted.forecast(steps=PredStep,
                                   alpha=Conf_int)
        f_values = forecast[0]
        f_conf_int = forecast[2]

        # Arranges the values to compute the weighted average
        pred_mat[:, 1:] = pred_mat[:, :-1]
        pred_mat[:, 0] = f_values
        conf_int_mat[:, 1:] = conf_int_mat[:, :-1]
        conf_int_mat[:, 0] = f_conf_int
        pred = np.diag(pred_mat)
        ci_sup = np.diag(conf_int_mat[:, :, 1])
        ci_inf = np.diag(conf_int_mat[:, :, 0])

        # Computes the predictions and the confidence intervals
        DF['pred'][x] = sum(pred * weights) / sum(weights * (pred != 0))
        DF['ci_sup'][x] = sum(ci_sup * weights) / sum((ci_sup != 0) * weights)
        DF['ci_inf'][x] = sum(ci_inf * weights) / sum((ci_inf != 0) * weights)

end = time.time()
print(end - start)


DF['Anomaly'] = (DF['ci_sup'] < DF['value']) | (DF['value'] < DF['ci_inf'])
dataplot = DF.copy()
dataplot['Anomaly_value'] = dataplot['Anomaly'] * dataplot['value']

# Plots

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly_value'], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("WEARIMA 100 Anomaly detection: Simple")
plt.show()


a = dt.datetime(2014, 1, 20)  # Fixes the start date
b = dt.datetime(2014, 3, 1)  # Fixes the start date

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'][a:b])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly_value'][a:b], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("WEARIMA 100 Anomaly detection: Detail Simple")
plt.show()

a = dt.datetime(2013, 12, 1)
b = dt.datetime(2014, 1, 15)

plt.figure(figsize=(10, 5))
plt.plot(dataplot['value'][a:b])
plt.plot(dataplot[dataplot.Anomaly != 0]['Anomaly_value'][a:b], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("WEARIMA 100 Anomaly detection: Detail Simple")
plt.show()
