#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:47:29 2019

@author: harshparikh
"""

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


n = 1000
X = []
Y = []
T = []

for i in range(0,n): 
    d0 = np.random.binomial(1,1/2)
    e0 = np.random.binomial(1,1/2)
    d = [d0]
    e = [e0]
    beta_d = [1,1]
    beta_e = [1,1]
    beta_y = [1,1]
    c = 2
    time = 10
    
    for j in range(1,time):
        ej = np.random.binomial(1,sigmoid(np.dot(beta_e,[np.mean(d)*np.mean(e),c])))
        e.append(ej)
        dj = np.random.binomial(1,sigmoid(np.dot(beta_d,[np.mean(d)*np.mean(e),c])))
        d.append(dj)
    
    yi = np.random.normal(loc=np.dot(beta_y,[np.mean(d)*np.mean(e),c]),scale=1)
    xi = [d,e]
    X.append(xi)
#    ti = d #continuous treat or multidimensional treatment
    


