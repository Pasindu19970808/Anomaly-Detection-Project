import os
import matplotlib as plt
import seaborn
import numpy as np
import pandas as pd
import re

# define path to data directory
file_path = os.path.join(os.getcwd(), 'KDD-Cup', 'data')
print(file_path)

file_to_load = os.path.join(file_path, os.listdir(file_path)[1])
print(file_to_load)

def split_data(file_path, index):
    file_name = os.listdir(file_path)[index]
    test_data_start_pt = int(re.findall(
        r'[0-9]*.txt', file_name)[0].split('.')[0])
    #data = pd.read_csv(os.path.join(file_path, os.listdir(file_path)[index]))
    data = [float(i.strip()) for i in open(os.path.join(file_path,os.listdir(file_path)[index])).readlines()]
    train_data = data[0:test_data_start_pt]
    test_data = data[test_data_start_pt+1:len(data)]

    return train_data, test_data

train, test = split_data(file_path, 1)
#train.head()
#test.head()




# # Above, initially set to only load first file into "data" and return training/testing data frames
#  in future could beextended to something like this
# for index in range(0,# files):
# #     train, test = split_data(file_path, index)
# #     train model
# #     test model
# #     append identified anomoly to list of anomoly items
