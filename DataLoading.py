import os
import matplotlib as plt
import seaborn
import numpy as np
import pandas as pd
import re

# define path to data directory
file_path = os.path.join(os.getcwd(),'data-sets' 'KDD-Cup', 'data')
print(file_path)

file_to_load = os.path.join(file_path, os.listdir(file_path)[1])
print(file_to_load)

#train, test = split_data(file_path, 1)
#train.head()
#test.head()


# def split_data(file_path, index):
#     file_name = os.listdir(file_path)[index]
#     test_data_start_pt = int(re.findall(
#         r'[0-9]*.txt', file_name)[0].split('.')[0])
#     data = pd.read_csv(os.path.join(file_path, os.listdir(file_path)[index]))
#     train_data = data[0:test_data_start_pt]
#     test_data = data[test_data_start_pt+1:len(data)]

#     return train_data, test_data


def split_data(file_path, index):

    file_name = os.listdir(file_path)[index]
    test_data_start_pt = int(re.findall(
        r'[0-9]*.txt', file_name)[0].split('.')[0])
    total_data = pd.read_csv(os.path.join(file_path, os.listdir(file_path)[index]))
    # Handle data formatting issue
    if total_data.shape[1] == 1:
        total_data = np.genfromtxt(os.path.join(file_path, os.listdir(file_path)[index]))
        total_data = pd.DataFrame(total_data)
    train_data = total_data[0:test_data_start_pt]
    test_data = total_data[test_data_start_pt+1:len(total_data)]
    return train_data, test_data, total_data, test_data_start_pt
    #scaler = StandardScaler()

    #scaler.fit(train_data)

    #train_scaled = scaler.transform(train_data)

    #test_scaled = scaler.transform(test_data)
