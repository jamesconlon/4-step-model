# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:52:56 2017

@author: James
"""

import numpy as np
import pandas as pd

data_raw = pd.read_csv('data.csv')

zones = data_raw['zone'].values
low_zone = min(zones)
high_zone = max(zones)

zone_id = []
zone_ref = []
for i in range(low_zone,high_zone+1):
    ref_ = np.where(zones == i)[0]
    
    if(len(ref_)>0):
        temp_ = [i,ref_]
        zone_ref.append(temp_)

variables = list(data_raw.columns.values)[1:] #ignore 'zone' variable
count = [len(arr[1]) for arr in zone_ref ]
                
def aggregate(data,zone_array,var_array):
    return_array = []
    for zone_reference in zone_array:
        indexes_ = zone_reference[1]
        
        sum_array = []
        for var in var_array:
            data_ = data[var].values[indexes_]
            sum_values = data_.sum()
            sum_array.append(sum_values)
            
        return_array.append(sum_array)         
    return(return_array)

agg = aggregate(data_raw,zone_ref,variables)
agg_df = pd.DataFrame(agg)
agg_df.columns=variables
agg_df['count'] = count
agg_df['zone'] = [arr[0] for arr in zone_ref]
agg_df = agg_df.set_index('zone')
out_csv = agg_df.to_csv('zone_totals.csv')

    
