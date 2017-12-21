# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:42:16 2017

@author: James
"""
import pandas as pd
import numpy as np
import networkx as nx

'''
Incrementally assigns flow to network
'''

g = nx.DiGraph()

nodes = [1,2,3,4]
#edges = [(1,2,10),(1,3,15),(2,3,3),(2,4,5),(3,4,4)] # format is (from,to,weight)
edges = [(1,2),(1,3),(2,3),(2,4),(3,4)]
edge_array = np.array([[edge[0],edge[1]] for edge in edges])
time_cost_init = [10,15,3,5,4]
g.add_nodes_from(nodes,labels=nodes)
g.add_edges_from(edges)

OD = pd.read_csv('OD.csv',index_col=0).values              
                
t_0 = [10,15,3,5,4]
C = [300,500,150,200,200]
volume = np.array([0]*5)  # for incremental method, we start with zero flows, and assume initial travel times
def VDF(link):#,volume):
    V_ = volume[link-1]
    t0_ = t_0[link-1]
    C1_ = C[link-1]
    
    t_1 = t0_/(1-V_/C1_)
    print(C1_)
    return(t_1)

def getEdgeIndex(pair):
    idx = np.where((edge_array[:,0] == pair[0]) & (edge_array[:,1] == pair[1]))[0][0] # the [0][0] makes it return a single integer
    return(idx)

OD_iter = OD *0.25 # 0.50

iter_order = np.array(      # Format: O,D, iteration 1 rank, iter 2 rank...
             [[1,2,1,3,5,3],
              [1,3,6,6,2,2],
              [1,4,2,4,1,6],
              [2,3,3,2,6,1],
              [2,4,5,5,4,4],
              [3,4,4,1,3,5]])
    
volume = np.array([0]*5)  
for iter_ in range(4):
    if(iter_ == 0):
        time_cost = time_cost_init
    else:
        time_cost = [VDF(link_) for link_ in range(1,6)]
    print(time_cost)
    iter_stack = np.column_stack((edge_array,time_cost))
    
    g_new = nx.DiGraph()
    new_edges = [(row[0],row[1],row[2]) for row in iter_stack]
    g_new.add_nodes_from(nodes,labels=nodes)
    g_new.add_weighted_edges_from(new_edges)
    for rank_ in range(6):
        idx_ = np.where(iter_order[:,2+iter_] ==(rank_+1))[0][0]
        O_ = iter_order[idx_,0]
        D_ = iter_order[idx_,1] 
        V_ = OD_iter[O_-1,D_-1] # need to subtract 1 since index started at zero
        path_ = nx.shortest_path(g_new,source=O_,target=D_, weight='weight')
        n_pairs = len(path_)-1 
        if(n_pairs>0):
            for i in range(n_pairs):
                temp_link = [path_[i],path_[i+1]]
                temp_edge_idx = getEdgeIndex(temp_link)
                volume[temp_edge_idx] += V_
    
nx.draw(g_new,node_size=600,node_color='b',edge_color='r',width=1,with_labels=True,font_color='w',font_size=12)
