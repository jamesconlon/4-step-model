# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 03:11:19 2017

@author: James
"""

import numpy as np
import pandas as pd
import itertools
from sklearn import linear_model

zone_data = pd.read_csv('zone_totals.csv')