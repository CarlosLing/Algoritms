import numpy as np
import pandas as pd


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
