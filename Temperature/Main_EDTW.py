import numpy as np
import pandas as pd
import pydtw as dtw
from pydtw.dtw import dtw1d

# Loading Data
Data = pd.read_csv('Data/Temperature/ambient_temperature_system_failure.csv',
                   parse_dates=True,
                   index_col='timestamp')

# Generate time series data:
# We will take 24 hours time series in order to detect the most anomalous time series based on magnitude and shape

from pydtw import dtw1d
import numpy as np
a = np.random.rand(10)
b = np.random.rand(15)
cost_matrix, alignmend_a, alignmend_b = dtw1d (a, b)