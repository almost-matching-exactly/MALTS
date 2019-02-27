#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:32:01 2018

@author: harshparikh
"""

import numpy as np
from . import distancemetriclearning as dml
import matplotlib.pyplot as plt

#Simulation 1
e = 0.1
na = [200,500,1000,1500,2000]
msea = []
for n in na:
    d = 20
    ct = np.random.uniform(0,1,n)
    beta = np.random.uniform(-5,5,size=20)
    X = []
    Y = []
    T = []
    for i in range(0,n):
        x = np.random.normal(0,5,size=20)
        outcome = np.dot(beta,x) + 8*np.random.binomial(1,0.5)
        if ct[i]<e:
            outcome = outcome + 5*np.random.binomial(1,0.1)
            T.append(1)
        else:
            T.append(0)
        Y.append(outcome)
        X.append(x)
     
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)
    #MALTS
    adknn_cobyla = dml.distance_metric_learning(20,discrete=False)
    optresult = adknn_cobyla.optimize(X,Y,T,numBatch=1)
    LstarC_cobyla = adknn_cobyla.Lc
    LstarT_cobyla = adknn_cobyla.Lt
    fLCcobyla = lambda y: LstarC_cobyla
    fLTcobyla = lambda y: LstarT_cobyla
    dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(X,X,Y,T,fLC=fLCcobyla,fLT=fLTcobyla)
    t_hat_cobyla = np.array(t_hat_cobyla)
    mse = np.sqrt( np.dot( t_hat_cobyla-8, t_hat_cobyla-8 ) )
    msea.append(mse)

plt.plot(msea)
