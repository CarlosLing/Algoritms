import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import datetime as dt
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from Functions.anomaly_arima import find_SARIMAX_parameters
# from Functions.anomaly_arima import ma_preprocess


Data = pd.read_csv('Data/Temperature/ambient_temperature_system_failure.csv',
                   parse_dates=True,
                   index_col='timestamp')
var_chosen = 'value'
ci_alpha = 0.01


# one_var = Data[var_chosen]
# plt.plot(one_var)


def ma_preprocess(data, window):
    l = len(data)
    Z = np.zeros(l)
    for x in range(l):
        Z[x] = np.mean(data[max(0, x - window // 2):min(l - 1, x + window // 2):])
    MA = pd.DataFrame(Z).set_index(data.index)
    return MA


one_var = ma_preprocess(Data[var_chosen], 4).rename(columns={0:var_chosen})
# The preprocess consists on a moving average
plt.plot(one_var)
# params = find_SARIMAX_parameters(data=one_var, range_param=2, range_param_season=2, seasonality=24)
# (1, 0, 1)
# (1, 1, 1, 24)

# model = SARIMAX(one_var,
#                order=params.param,
#                seasonal_order=params.param_season,
#                enforce_stationarity=False,
#                enforce_invertibility=False)
# results = model.fit()

# Fits the chosen SARIMA model
model = SARIMAX(one_var,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

# Shows the results: Summary and Plots
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Predicts the variable values and the confidence intervals
pred = results.get_prediction(start=pd.to_datetime('2013-08-01'), dynamic=False)
pred_ci = pred.conf_int(alpha=ci_alpha)

# Creates a Data Frame to plot the results
x = pd.DataFrame(one_var)
plot_df = pd.concat([x, pred_ci], axis=1)

plot_df['Flag_Anomaly'] = (plot_df[var_chosen] > plot_df['upper ' + var_chosen]) | \
                          (plot_df[var_chosen] < plot_df['lower ' + var_chosen])
plot_df['Anomaly'] = plot_df[var_chosen] * plot_df['Flag_Anomaly']

###
# Plot Routine

fig = plt.figure(figsize=(10, 5))
plt.plot(plot_df[var_chosen])
plt.fill_between(x=plot_df.index, y1=plot_df['upper ' + var_chosen], y2=plot_df['lower ' + var_chosen], alpha=0.4)
plt.plot(plot_df[plot_df.Anomaly != 0]['Anomaly'], 'ro')
plt.xlabel('Time')
plt.ylabel(var_chosen)
plt.legend()
plt.title("ARIMA-Based Anomaly detection")

# Plot Routine

a = dt.datetime(2014, 1, 20)  # Fixes the start date
b = dt.datetime(2014, 3, 1)  # Fixes the start date

plot_dfab = plot_df[a:b]

plt.figure(figsize=(10, 5))
plt.plot(plot_dfab[var_chosen])
plt.fill_between(x=plot_dfab.index, y1=plot_dfab['upper ' + var_chosen], y2=plot_dfab['lower ' + var_chosen], alpha=0.4)
plt.plot(plot_dfab[plot_dfab.Anomaly != 0]['Anomaly'], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("ARIMA-Based Anomaly detection： Detail Simple")
plt.show()

a = dt.datetime(2013, 12, 1)
b = dt.datetime(2014, 1, 15)

plot_dfab = plot_df[a:b]

plt.figure(figsize=(10, 5))
plt.plot(plot_dfab[var_chosen])
plt.fill_between(x=plot_dfab.index, y1=plot_dfab['upper ' + var_chosen], y2=plot_dfab['lower ' + var_chosen], alpha=0.4)
plt.plot(plot_dfab[plot_dfab.Anomaly != 0]['Anomaly'], 'ro')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title("ARIMA-Based Anomaly detection： Detail Simple")
plt.show()
