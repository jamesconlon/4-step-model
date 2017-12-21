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

variables = list(zone_data.columns.values)[2:-4]

data = zone_data[variables].values

pca = PCA(n_components=3)
pca.fit(data)
comp = pca.components_

pca_df = pd.DataFrame(comp)
pca_df.columns = variables

'''
i_ = np.identity(11)
trans_ = pca.transform(i_)
pca_df = pd.DataFrame(trans_)

'''

