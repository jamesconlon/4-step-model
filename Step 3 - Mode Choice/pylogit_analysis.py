# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:14:54 2017

@author: James
"""

import pandas as pd
import numpy as np
import pylogit as pl
from collections import OrderedDict

##############################################################################
raw_data = pd.read_csv('mode_choice_data_original.csv')
raw_data_len = len(raw_data)
missing_values = pd.read_csv('missing_values.csv',header=0,index_col='code') # missing_values.loc['occ'].val
modes = ['a-dr','a-pass','transit','rail','walk'] # for modes: 1 = chosen, 0 = available/unchosen, -1 = unavailable/unchosen
choice_stack = raw_data[modes].values
choice_val = np.array([np.where(row==1)[0][0]for row in choice_stack])
choice_desc =[modes[val] for val in choice_val]
specification = OrderedDict()
names = OrderedDict()
a_dr_idx = np.where((raw_data[modes[0]] == 0) |(raw_data[modes[0]] == 1))[0]
a_dr_avail = np.zeros(len(raw_data))
a_dr_avail[a_dr_idx] =1
a_pass_idx = np.where((raw_data[modes[1]] == 0) |(raw_data[modes[1]] == 1))[0]
a_pass_avail = np.zeros(len(raw_data))
a_pass_avail[a_pass_idx] =1
transit_idx = np.where((raw_data[modes[2]] == 0) |(raw_data[modes[2]] == 1))[0]
transit_avail = np.zeros(len(raw_data))
transit_avail[transit_idx] =1
rail_idx = np.where((raw_data[modes[3]] == 0) |(raw_data[modes[3]] == 1))[0]
rail_avail = np.zeros(len(raw_data))
rail_avail[rail_idx] =1      
walk_idx = np.where((raw_data[modes[4]] == 0) |(raw_data[modes[4]] == 1))[0]
walk_avail = np.zeros(len(raw_data))
walk_avail[walk_idx] =1     
avail_stack = np.column_stack((a_dr_avail,a_pass_avail,transit_avail,rail_avail,walk_avail))
modes_avail = [str_+'_AV' for str_ in modes]
for i_,avail in enumerate(modes_avail):
    raw_data[avail] = avail_stack[:,i_]
raw_data['CHOICE'] = np.array(choice_val+1)
##############################################################################
# MODE CHOICES
availability_variables = {1: 'a-dr_AV',
                          2: 'a-pass_AV',
                          3: 'transit_AV', 
                          4: 'rail_AV', 
                          5: 'walk_AV'}
##############################################################################



raw_data['female_int'] = (raw_data['female'] == 1).astype(int)

raw_data['trdist_split'] = (raw_data['trdist'] >26).astype(int)  

raw_data['walk_no_car'] = ((raw_data['trdist']<=5)*(raw_data['a-dr_AV'] ==0)).astype(int)

raw_data['typical_office'] = ((raw_data['jbsite'] ==1)*(raw_data['wkweek'] ==1)).astype(int)   

raw_data['wtime'] = raw_data['trdist']/5*60

raw_data['free_int'] = (raw_data['free'] ==1).astype(int)

raw_data['veh_available'] = ((raw_data['nveh']>=1)*(raw_data['nveh']!=9)).astype(int)
raw_data['veh_worker'] = (raw_data['nveh'] < raw_data['nwkrs']).astype(int)


raw_data['parking_adj'] = (raw_data['free'] != 1).astype(int)*(raw_data['mcost'].values)
raw_data['no_car'] = (raw_data['nveh']==0).astype(int)

raw_data['house'] = (raw_data['dweltp']==1).astype(int)
raw_data['zero_fare'] = (raw_data['CHOICE']*0).astype(int)

age_int = 40
raw_data['age_boolean'] = (raw_data['age']>=age_int).astype(int)

raw_data['hhsize_boolean']=(raw_data['hhsize']>=2).astype(int)

raw_data['part-time'] = (raw_data['empst']==2).astype(int)
raw_data['part-time_no_veh'] = ((raw_data['empst']==2)*(raw_data['a-dr_AV']==0)).astype(int)

# aggregating varying variables by mode type
alt_varying_variables = {
                          u'wait time': dict([(3, 'twait'),
                                              (4, 'r-wait')]),

    
                          u'fare': dict([(3, 't-fare'),
                                                (4, 'r-fare'),
                                                ]),

                          u'access time': dict([(3, 't-accw'),
                                                (4, 'r-acca'),]),
    
                            u'ivtt': dict([(1, 'aivtt'),
                                                (2, 'aivtt'),
                                                (3, 'tivtt'),
                                                (4, 'rivttg'),
                                                (5, 'wtime')]),
    
    
    }
   
columns = raw_data.columns.tolist()
custom_alt_id = 'mode_ID'
choice_column = 'CHOICE_'
raw_data[choice_column] = np.array(choice_val+1)
obs_id_column = "custom_id"
raw_data[obs_id_column]=np.array(raw_data.index+1)
used_cols = ['female','age','dlic','wkweek','hinc','trdist','hhsize']
def missingIndexes(stack=raw_data,col_array=used_cols,missing_array=missing_values):
    missing_idx = []
    for col in col_array:
        if col in list(missing_array.index):
            val_ = missing_values.loc[col].val
            arr_ = stack[col].values
            idx_ = list(np.where(arr_ == val_)[0])
            missing_idx.extend(idx_)
    return(np.unique(missing_idx))
    

def missingToUsable(missing,len_):
    return_arr = np.ones(len_)
    return_arr[list(missing)]=0
    return_idx = np.where(return_arr ==1)[0]
    return(return_idx)
    

    
missing = missingIndexes()
usable = missingToUsable(missing,len(raw_data))



long_data = pl.convert_wide_to_long(raw_data, 
                                           columns, 
                                           alt_varying_variables, 
                                           availability_variables, 
                                           obs_id_column, 
                                           choice_column,
                                           new_alt_id_name=custom_alt_id)
    
missing_long = missingIndexes(stack=long_data)
usable_long = missingToUsable(missing_long,len(long_data))
def addVariable(var_name,modes,desc_array,names=names,specification=specification):
    specification[var_name] = modes
    names[var_name] = desc_array 
        
        
        
def getModel(alt,obs,choice,spec = specification,names = names,type_="MNL",data=raw_data):
    model = pl.create_choice_model(data=data,
                                     alt_id_col=alt,
                                     obs_id_col=obs,
                                     choice_col=choice,
                                     specification=specification,
                                     model_type=type_,
                                     names=names)
    return(model)
        

# ADD VARIABLES HERE
##############################################################################
addVariable('intercept',[2,3,4,5],['I2','I3','I4','I5'])
addVariable('female_int',[1,3],['female1','female3'])
addVariable('free_int',[1,4],['free1','free4'])
#addVariable('parking_adj',[1,4],['free1','free4'])
addVariable('dlic',[2],['dlic2'])
addVariable('walk_no_car',[5],['walk_no_car5'])
addVariable('trdist_split',[3,4],['dist3','dist4'])
addVariable('typical_office',[3],['office3'])
addVariable('trdist',[1,5],['dist1','dist5'])
addVariable('hinc',[1,4],['inc1','inc4'])
addVariable('fare',[[3,4]],['fare'])
addVariable('hhsize_boolean',[2],['hhsize2'])
addVariable('pinc',[2],['pinc2'])
addVariable('house',[2],['house2'])
addVariable('part-time',[2],['pt2'])
##############################################################################


long_data_usable = long_data.iloc[list(usable_long),:]

model = getModel(custom_alt_id,obs_id_column,choice_column,data=long_data_usable)

model.fit_mle(np.zeros(22))
model.get_statsmodels_summary()

model.print_summaries()
        
