import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Define split data

def calculate_prominence_without_smoothing(anomaly_scores):
    #In this case we define anomaly score as simply the division between the highest anomaly score and the second highest
    highest_anomaly_score = sorted(zip(range(anomaly_scores.shape[0]),anomaly_scores), key = lambda x : x[1], reverse = True)[0][1]
    second_highest_anomaly_score = sorted(zip(range(anomaly_scores.shape[0]),anomaly_scores), key = lambda x : x[1], reverse = True)[1][1]
    return highest_anomaly_score/second_highest_anomaly_score

def split_data(file_path, index):
    file_name = os.listdir(file_path)[index]
    test_data_start_pt = int(re.findall(
        r'[0-9]*.txt', file_name)[0].split('.')[0])
    total_data = pd.read_csv(os.path.join(file_path, os.listdir(file_path)[index]))
    train_data = total_data[0:test_data_start_pt]
    test_data = total_data[test_data_start_pt+1:len(total_data)]

    return train_data, test_data, total_data, test_data_start_pt

file_path = os.path.join(os.getcwd(), 'KDD-Cup', 'data')
file_name = os.listdir(file_path)[1]
file_to_load = os.path.join(file_path, file_name)
train, test, total, threshold = split_data(file_path, 1)




def Detect_AutoRegression(train,test,threshold):
    train = np.array(train).reshape(-1,1)
    test = np.array(test).reshape(-1,1)
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    window = 100
    model = AutoReg(train_scaled,lags = window)
    model_fit = model.fit()
    coef = model_fit.params
    #to make the first m predictions in the test data, we need the last 100 data from train dataset
    history = train_scaled[len(train_scaled) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    residual_err = list()

    for t in range(len(test_scaled)):
        length = len(history)
        #finding the past 100 points to predict the next point as prediction requires
        #applying coef from model_fit.params to the lag values
        lag = [history[i] for i in range(length - window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window - d - 1]
        new_history = test_scaled[t]
        predictions.append(yhat)
        history.append(new_history)
        residual_err.append(np.abs(new_history-yhat)[0])
    #smoothed_anomaly_scores = smoothing(np.array(residual_err),m)
    #prominence,outlier_position = calculate_prominence(smoothed_anomaly_scores,m)
    #calculate outlier position without smoothing

    
    outlier_position = np.argsort(np.array(residual_err))[-1]
    prominence = calculate_prominence_without_smoothing(np.array(residual_err))
    outlier_position = outlier_position + threshold
    print("Predicted anamoloy from AR algorithm  is located at location: " + str(outlier_position))
    anomaly_positions_without_smoothing = tuple(zip(range(len(residual_err)),residual_err))
    scores = sorted(anomaly_positions_without_smoothing, key = lambda x:x[1],reverse=True)
    first_outlier_position = scores[0][0] + threshold
    return first_outlier_position, pd.Series(np.array(residual_err)),prominence


Detect_AutoRegression(train,test,threshold)