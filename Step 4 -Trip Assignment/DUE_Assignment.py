# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:42:16 2017

@author: James
"""

import pandas as pd
import numpy as np
import networkx as nx


'''
Calculates a DUE for the network
'''

nodes = [1,2,3,4]
edges = [(1,2),(1,3),(2,3),(2,4),(3,4)]
links = [1,2,3,4,5]
edge_array = np.array([[edge[0],edge[1]] for edge in edges])
time_cost_init = [10,15,3,5,4]

OD = pd.read_csv('OD.csv',index_col=0).values              
                
t_0 = [10,15,3,5,4]
C = [300,500,150,200,200]
# start with volumes from incremental assignment
volume_init = np.array([175,125,73,98,150]) # incremental assignment (4 iterations)
#volume_init = np.array([165,132,81,81,165])  # incremental assignment (3 iterations)
#volume_init = np.array([200,100,100,100,150]) # incremental assignment (2 iterations)
#volume_init = np.array([300,0,150,150,100]) # all-or-nothing
def VDF(link,v_='init'):
    if(v_ == 'init'):
        V_ = volume_init[link-1]
    else:
        V_ = v_
    t0_ = t_0[link-1]
    C1_ = C[link-1]
    t_1 = t0_/(1-V_/C1_)
    return(t_1)

def getEdgeIndex(pair):
    idx = np.where((edge_array[:,0] == pair[0]) & (edge_array[:,1] == pair[1]))[0][0]
    return(idx)

def integrated_VDF (link,v_):
    t0_ = t_0[link-1]
    C1_ = C[link-1]
    
    integral = -1*t0_*C1_*(np.log(C1_-v_) - np.log(C1_ - 0))
    return(integral)

def allVDF(v_arr):
    
    VDFs = [VDF(_+1,v_) for _,v_ in enumerate(v_arr)]
    return(VDFs)

nonzero_OD = np.array(
             [[1,2],
              [1,3],
              [1,4],
              [2,3],
              [2,4],
              [3,4]])

# All-or-nothing assignment

def allOrNothingAssignment(V_array):
    new_link_flows = np.zeros(5)
    for pair in nonzero_OD:
        g = nx.DiGraph()
        g.add_nodes_from(nodes,labels=nodes)
        time_cost = allVDF(V_array)
        weighted_edge_array = np.column_stack((edge_array,time_cost))
        weighted_edges = [(row[0],row[1],row[2]) for row in weighted_edge_array]
        g.add_weighted_edges_from(weighted_edges)

        O_, D_ = pair[0], pair[1]
        V_ = OD[O_ - 1, D_ -1]
        shortest_path = nx.shortest_path(g,source=O_,target=D_,weight='weight')
        n_pairs = len(shortest_path)-1
        for i in range(n_pairs):
            temp_link = [shortest_path[i],shortest_path[i+1]]
            temp_edge_idx = getEdgeIndex(temp_link)
            new_link_flows[temp_edge_idx] += V_
    return(new_link_flows)


lambda_array = [0,0.25,0.5,0.75,1.0]
def flowComparison(V,V_prime):
    V = np.array(V)
    V_prime = np.array(V_prime)
    def V_double_prime(lambda_):
        vpp = lambda_*V + (1-lambda_)*V_prime
        return(vpp)
    
    compare_stack = []
    for l in lambda_array:
        S = allVDF(V_double_prime(l))
        compare_stack.append(S)
        
    compare_stack = np.array(compare_stack)
    lambda_sums = np.array([row.sum() for row in compare_stack])
    lambda_min_idx = np.argmin(lambda_sums)
    lambda_choice = lambda_array[lambda_min_idx]
    vpp = V_double_prime(lambda_choice)
    return(vpp,vpp == V)

v_double_prime = np.zeros(5)
for i in range(10):
    if(i == 0):
        v = volume_init
    else:
        v = v_double_prime
    v_prime = allOrNothingAssignment(v)
    v_double_prime, equal_array = flowComparison(v,v_prime)
    print(equal_array,v_double_prime)

