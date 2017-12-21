# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:19:48 2017

@author: James
"""
import numpy as np
import pandas as pd
import itertools
from sklearn import linear_model

zone_data = pd.read_csv('zone_totals.csv')

variables = list(zone_data.columns.values)[1:-4]

it_arr = []
for i in range(1,len(variables)+1):
    it_ = itertools.combinations(variables,i)
    it_arr.extend(it_)
    
    
def r_adjust(R2,k,n):
    if(n==1):
        return(R2)   
    r_adj = (R2-(k/(n-1)))*((n-1)/(n-k-1))
    return(r_adj)    

def sum_squares(predict,y):
    stack_ = np.column_stack((predict,y))
    squares = [(row[0] - row[1])**2 for row in stack_]

    return(sum(squares))

    
def reg(data,var,y_str):
    X = data[var].values
    y = data[y_str].values
    reg = linear_model.LinearRegression()
    model = reg.fit(X,y)
    r_square = model.score(X,y)        
    k = len(var)
    n = len(X)
    r_square_adjust = r_adjust(r_square,k,n)
    coef = model.coef_
    intercept = model.intercept_
    predict = model.predict(X)
    ssq = sum_squares(predict,y)
    
    return(predict,r_square,r_square_adjust,coef,intercept,ssq)
    
predict = []
r2 = []
r2_adj = []
coef = []
intercept = []
ssq = []
n_var = []


for it_ in it_arr:
    
    predict,r_square_,r_square_adjust_,coef_,intercept_,ssq_ = reg(zone_data,list(it_),'nwork')
    n_var.append(len(list(it_)))
    r2.append(r_square_)
    r2_adj.append(r_square_adjust_)
    coef.append(coef_)
    intercept.append(intercept_)   
    ssq.append(ssq_)
  
stack = np.column_stack((n_var,r2_adj,intercept,ssq))
out_df = pd.DataFrame(stack)
out_df.columns = ['n_variables','adjusted_r2','intercept','sum of squares']
out_df['vars']=it_arr
out_df.to_csv('iterative_data.csv')


            