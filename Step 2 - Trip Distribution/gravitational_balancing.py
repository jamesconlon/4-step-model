# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 01:06:41 2017

@author: James
"""

import pandas as pd
import numpy as np

'''
input files:
    base data: 'table1.csv'
    travel times: 'table2.csv'
    future data: 'table3.csv'
'''


travel_time_raw = pd.read_csv('table2.csv')
travel_time = np.array(travel_time_raw)
travel_time = np.delete(travel_time,0,axis=1)

n_zones = 5
acceptable_error = 5 / 100 # (5% error)

zone = [i for i in range(1,n_zones+1)]
zone_idx = list(np.array(zone)-1)

future_data = pd.read_csv('table3.csv')
Oi_new_raw = future_data['Origins'].values[:-1] 
Oi_new = np.array(np.matrix(Oi_new_raw).T)

Dj_new_raw = future_data['Dest.'].values[:-1]
Dj_new = np.array(np.matrix(Dj_new_raw))

base_data_raw = pd.read_csv('table1.csv')[zone].values
base_data = np.delete(base_data_raw,-1,axis=0)

def error(factor):
    return(np.round((np.abs(1-factor)),3)*100)

# This is the impedance function, set to exponential, but can be changed
def f_ij(a):
    return(np.exp(-.064*a))

def sum_O(array,index):
    O_i = [[array[idx,:].sum()] for idx in index]
    return(np.matrix(O_i))

def sum_D(array,index):
    D_j = [array[:,idx].sum() for idx in index]
    return(np.matrix(D_j))

T_ij = base_data

impedance_vector = np.vectorize(f_ij)
impedance = impedance_vector(travel_time)

O_i = Oi_new
D_j = Dj_new

D_j_list = list(D_j[0])
O_i_list =  [i_[0] for i_ in O_i]
iteration = 0
acceptable = False

while (acceptable == False):
    iteration = iteration + 1
 
    if(iteration == 1):
        D_j_adjusted = D_j_list
        T_ij_new = np.ones((5,5))
    
    for i in range(0,n_zones):
        O_i_ = O_i[i][0]
        for j in range(0,n_zones):
            D_j_adjusted_ = D_j_adjusted[j]
            travel_time_ = travel_time[i][j]
            impedance_ = f_ij(travel_time_)
            numerator = O_i_ * D_j_adjusted_ * impedance_
            
            D_list = T_ij_new[:,j] * impedance_
            denominator = D_list.sum()
            T_ = numerator / denominator
            T_ij_new[i][j] = T_   
    R_j = []
    for j in range(0,n_zones):
        D_ = D_j_list[j]
        sum_T = np.array(sum(T_ij_new[:,j]))
        R_j.append(D_ / sum_T)

    R_j_compare = [np.abs(R - 1) for R in R_j]
    print('\n iteration {}:\n'.format(iteration),T_ij_new)
    print('R: ',R_j_compare)    
    if(max(R_j_compare) < acceptable_error):
        print('finished, took {} iterations'.format(iteration))
        break
    else:
        D_j_adjusted = np.array(D_j_adjusted) * np.array(R_j)
