#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Distance with 0-1 values
@author: dylan
"""

import numpy as np
import math 
m, n = np.random.randint(100), np.random.randint(100)
X = np.zeros((n,m))
X
for i in range(n):
    for j in range(m):
        X[i,j] = np.random.binomial(1, .1)

X.shape
X
#%%
class matrix_01:
    def __init__(self, data, dist_type = 'L_infty', density = 'dense' ):
        self.data = data
        self.dist_type = dist_type
        self.density = density    
        self.dist_matrix = np.zeros(data.shape)
    
    # for dense matrices

    def create_dist_matrix(self):
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.data[i,j] == 1:
                    continue
                self.dist_matrix[i,j] = self._nearest_pt_dense((i,j), self.dist_type)
        return self.dist_matrix

    def _nearest_pt_dense(self, datum_index, dist_type):
        if dist_type == 'L_infty':
            dat_range = 0
            distance = 0
            while not distance:
                dat_range += 1
                for i in range(max(0, datum_index[0] - dat_range), min(self.data.shape[0], datum_index[0] + dat_range +1)):
                    for j in range(max(0, datum_index[1] - dat_range), min(self.data.shape[1], datum_index[1] + dat_range +1)):
                        if self.data[i,j] == 1:
                            distance = dat_range
                            break
        elif dist_type == 'L_2':
            dat_range = 0
            distance = 0
            while not distance:
                dat_range += 1
                for k in range(1, dat_range+1):
                    for sign in ((1,1), (1,-1), (-1,1), (-1,-1)):                       
                        if sign[0]*sign[1] == 1:
                            # Boundary Conditions #
                            if datum_index[0]+sign[0]*k < 0 or datum_index[0]+sign[0]*k >= self.data.shape[0]:
                                continue
                            if datum_index[1]+sign[1]*(dat_range-k) < 0 or datum_index[1]+sign[1]*(dat_range-k) >= self.data.shape[1] :
                                continue
                            # Identify nearest 1 #
                            if self.data[datum_index[0]+sign[0]*k, datum_index[1]+sign[1]*(dat_range-k)] == 1:
                                distance = dat_range
                                break
                        else:
                            # Boundary Conditions #
                            if datum_index[0]+sign[0]*(dat_range-k) < 0 or datum_index[0]+sign[0]*(dat_range-k) >= self.data.shape[0]:
                                continue
                            if datum_index[1]+sign[1]*k < 0 or datum_index[1]+sign[1]*k >= self.data.shape[1] :
                                continue
                            # Identify nearest 1 #
                            if self.data[datum_index[0]+sign[0]*(dat_range-k), datum_index[1]+sign[1]*k] == 1:
                                distance = dat_range
                                break
        else:
            print("Please enter L_2 or L_infty as distance types.")
        return distance

#    def _bdry_check(datum_index, sign, k, dat_range):
#        return datum_index[0]+sign[0]*k < 0 or datum_index[0]+sign[0]*k > self.data.shape[0] or datum_index[1]+sign[1]*(dat_range-k) < 0 or datum_index[1]+sign[1]*(dat_range-k) > self.data.shape[1] 
    
    #def _sparse_nearest_pt()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
    
    
    
    
    
    
    
    