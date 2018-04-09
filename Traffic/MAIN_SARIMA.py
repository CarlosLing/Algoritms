import time
import warnings
import itertools
import pandas as pd
import datetime as dt  # Imports dates library

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt


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


def find_pdq(data, d_lim=2, range_pq=3):
    # Data must be a time series
    res = adfuller(data)
    is_stationary = res[1] < 0.01  # Checks stationarity for the first time
    d = 0
    # Differenciates until stationarity
    while not is_stationary and d < d_lim:
        data = data.diff().dropna()
        res = adfuller(data)
        is_stationary = res[1] < 0.01
        d += 1
    [p, q] = pq_arima(data, range_pq)
    return [p, d, q]


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
# DF.to_csv('Data/Traffic/sample_2.csv')
###
# End of loading and arranging data
###

# Chooses a variable
var_chosen = 'occupancy_t4013'
one_var = DF[var_chosen]
seasonality = 24
min_cor = 0.2
ci_alpha = 0.05

# We want to choose the most correlated variables to use as exogenous variables
correl = abs(DF.corr()[var_chosen]).sort_values(ascending=False)
correl_var = correl[correl > min_cor][1:].index

# Create function to determine d parameter to stationarity
[p, d, q] = find_pdq(one_var, range_pq=3)
[P, D, Q] = find_pdq(one_var.diff(seasonality).dropna(), range_pq=3)

# Fits the chosen SARIMA model
model = SARIMAX(endog=one_var,
                exog=DF.drop(columns=var_chosen)[correl_var],
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonality),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(maxiter=200)

# Shows the results: Summary and Plots
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Predicts the variable values and the confidence intervals
pred = results.get_prediction(start=pd.to_datetime('2015-09-11'), dynamic=False)
pred_ci = pred.conf_int(alpha=ci_alpha)

# Creates a Data Frame to plot the results
x = pd.DataFrame(one_var)
plot_df = pd.concat([DF, pred_ci], axis=1)

plot_df['Flag_Anomaly'] = (plot_df[var_chosen] > plot_df['upper ' + var_chosen]) | \
                          (plot_df[var_chosen] < plot_df['lower ' + var_chosen])
plot_df['Anomaly'] = plot_df[var_chosen] * plot_df['Flag_Anomaly']

###
# Plot Routine

fig = plt.figure(figsize=(10, 5))
plt.plot(plot_df[var_chosen])
if len(correl_var) > 0:
    for var in correl_var:
        plt.plot(plot_df[var])
plt.fill_between(x=plot_df.index, y1=plot_df['upper ' + var_chosen], y2=plot_df['lower ' + var_chosen], alpha=0.4)
plt.plot(plot_df[plot_df.Anomaly != 0]['Anomaly'], 'ro')
plt.xlabel('Time')
plt.ylabel(var_chosen)
plt.legend()
plt.title("SARIMAX-Based Anomaly detection")
