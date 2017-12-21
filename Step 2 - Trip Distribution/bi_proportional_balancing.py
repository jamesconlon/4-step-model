# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:14 2017

@author: James
"""

import pandas as pd
import numpy as np

'''
takes input tables for base and projected data and 
calculates a biproportionally balanced OD matrix
'''

#############################################
future_data_string = 'projected.csv'
base_data_string = 'base_data.csv'
n_zones = 5  #number of zones (NxN matrix)
acceptable_error = 5 #acceptable percent error
#############################################

zone = [i for i in range(1,n_zones+1)]
zone_idx = list(np.array(zone)-1)

future_data = pd.read_csv(future_data_string)
Oi_new_raw = future_data['Origins'].values[:-1] 
Oi_new = np.array(np.matrix(Oi_new_raw)) 

Dj_new_raw = future_data['Dest.'].values[:-1]
Dj_new = np.array(np.matrix(Dj_new_raw).T)
     
base_data_raw = pd.read_csv(base_data_string)[zone].values
base_data = np.delete(base_data_raw,-1,axis=0) 


def error(factor):
    return(np.round((np.abs(1-factor)),3)*100)

def scalarArray(ratio_list):
    temp_scalar_arr = np.array([[ratio] for ratio in ratio_list])
    return(temp_scalar_arr)

def sum_D(array,index):
    D_j = [[array[idx,:].sum()] for idx in index]
    return(np.matrix(D_j))

def sum_O(array,index):
    O_i = [array[:,idx].sum() for idx in index]
    return(np.matrix(O_i))

T_ij = base_data
iteration = 0
acceptable = False

while(acceptable == False):
    iteration = iteration + 1 
    print('Iteration {}'.format(iteration))

    O_i = np.array(sum_O(np.array(T_ij),zone_idx))
    factor_O = Oi_new / O_i

    T_ij = T_ij * factor_O
    print(T_ij,'\n')

    D_j = np.array(sum_D(T_ij,zone_idx))    
    factor_D = Dj_new / D_j

    T_ij = T_ij * factor_D
    print(T_ij,'\n')

    factor_O_np = np.array(factor_O)[0,:]
    factor_D_np = np.array(factor_D)[:,0]

    O_error = [error(f) for f in factor_O_np]
    D_error = [error(f) for f in factor_D_np]
    
    print( 'O Error:', ['{}%'.format(round(err,3)) for err in O_error])
    print( 'D Error:', ['{}%'.format(round(err,3)) for err in D_error])
    print('\n')

    if((max(O_error) < acceptable_error) and (max(D_error) < acceptable_error)):
        acceptable = True
        print('finished')
        out_df = pd.DataFrame(T_ij).to_csv('balanced_OD.csv')
