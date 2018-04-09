import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def find_SARIMAX_parameters(data, range_param, range_param_season, seasonality):
    class SARIMAX_Parameters:
        param = (0, 0, 0)
        param_season = (0,0,0,0)
        seasonality = 0

    # Define the p, d and q parameters to take any value between 0 and range_param
    p = d = q = range(0, range_param)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Define the p, d and q parameters to take any value between 0 and range_param
    p = d = q = range(0, range_param_season)

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]

    #   warnings.filterwarnings("ignore") # specify to ignore warning messages
    # Chooses the adequated parameters attending to the AIC

    aic = -1
    c_param = (0, 0, 0)
    c_param_seasonal = (0, 0, 0)
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(data,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit(disp=False)
                if aic == -1 or aic > res.aic:
                    print('ARIMA{}x{}24 - AIC:{}'.format(param, param_seasonal, res.aic))
                    aic = res.aic
                    c_param = param
                    c_param_seasonal = param_seasonal
            except:
                continue

    param_def = SARIMAX_Parameters()
    param_def.param = c_param
    param_def.param_season = c_param_seasonal
    param_def.seasonality = seasonality
    return param_def

def ma_preprocess(data, window):
    l = len(data)
    Z = np.zeros(l)
    for x in range(l):
        Z[x] = np.mean(data[max(0, x - window // 2):min(l - 1, x + window // 2):])
    MA = pd.DataFrame(Z).set_index(data.index)
    return MA
