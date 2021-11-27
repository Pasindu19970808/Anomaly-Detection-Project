import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import stumpy
from sklearn.preprocessing import StandardScaler
from dao import DataLoading
from matplotlib.patches import Rectangle
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA


train,test,test_data_start_pt = DataLoading.split_data(file_path = DataLoading.file_path,index = 1)
train = np.array(train).reshape(-1,1)
test = np.array(test).reshape(-1,1)
scaler = StandardScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

mp = stumpy.stump(test_scaled.reshape(test_scaled.shape[0],),100,train_scaled.reshape(train_scaled.shape[0],),ignore_trivial=False,normalize=False)

def smoothing(anomaly_data,m):
    start_point = m
    end_point_for_smoothing = anomaly_data.shape[0] - 1
    smoothed_anomaly_scores = np.zeros(shape = (anomaly_data.shape[0]))
    for i in range(start_point,end_point_for_smoothing):
        smoothed_anomaly_scores[i] = np.mean(anomaly_data[i - m:i+1])
    return smoothed_anomaly_scores