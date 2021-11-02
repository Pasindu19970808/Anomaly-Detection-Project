import numpy as np

def calculate_difference(timeseries, interval = 1):
    d1 = np.array(timeseries[0:-1])
    d2 = np.array(timeseries[interval:])
    return d2 - d1