# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:12:58 2017

@author: James
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
from sklearn.decomposition import PCA

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


predict,r_square_,r_square_adjust_,coef_,intercept_,ssq_ = reg(zone_data,['nftw'],'nwork')








