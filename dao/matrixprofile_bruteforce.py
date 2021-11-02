from re import sub
import numpy as np
import math

#https://towardsdatascience.com/the-matrix-profile-e4a679269692

results_dict = {}

def calculate_matrix_profile(time_series,m):
    n = len(time_series)
    for i in range(n - m + 1):
        ref_list = np.array(time_series[i:i+m])
        minimum_dist = float("+inf")
        for j in range(n - m + 1):
            if i == j:
                continue
            else:
                subsequence = np.array(time_series[j:j+m])
                euclidean_distance = math.sqrt(np.sum((ref_list - subsequence)**2))
                if euclidean_distance < minimum_dist:
                    minimum_dist = euclidean_distance
                    results_dict[i] = minimum_dist
    return results_dict
