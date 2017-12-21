# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:52:56 2017

@author: James
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import itertools
zone_data = pd.read_csv('zone_totals.csv')


x_str = ['npers','nveh','nftw','nwah','nstud','n65+']
X = zone_data[x_str].values
y = zone_data['nwork'].values
             
             
reg = linear_model.LinearRegression()
model = reg.fit(X,y)
r_square = model.score(X,y)

def r_adjust(R2,k,n):
    if(n==1):
        return(R2)
    
    r_adj = (R2-k/(n-1))*((n-1)/(n-k-1))
    return(r_adj)

k = len(x_str)
n = len(X)

r_square_adjust = r_adjust(r_square,k,n)

coef = model.coef_
intercept = model.intercept_

predict = model.predict(X)


def sum_squares(predict,y):
    stack_ = np.column_stack((predict,y))
    squares = [(row[0] - row[1])**2 for row in stack_]
    #error = np.sqrt(squares)
    #se = error/np.sqrt(n)
    return(sum(squares))

ssq = sum_squares(predict,y)


test = list(itertools.combinations(x_str,2))
print(test)





    