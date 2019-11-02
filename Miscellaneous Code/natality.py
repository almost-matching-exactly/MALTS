#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:28:28 2018

@author: harshparikh
"""

import pandas as pd
import numpy as np
import distancemetriclearning as dml

df = pd.read_csv('../../../SudeepaNatalityData/Natality_small.csv')
df = df.sample(frac=1)
df_dummies = pd.get_dummies(df,drop_first=True)
cols = list(df_dummies.columns)
l = ['FiveMinAPGARScore','CigaretteRecode_Y']+list(filter(lambda x: 'Flag' in x, cols))
X = np.array(df_dummies[ df_dummies.columns.difference( l) ] ) 
Y = np.array(df_dummies['FiveMinAPGARScore'])
T = np.array(df_dummies['CigaretteRecode_Y'])

Xb,Yb,Tb = X[:1000,:],Y[:1000],T[:1000]
n,p = Xb.shape
adknn = dml.distance_metric_learning(p,discrete=False)
optresult = adknn.optimize(Xb[:,:],Yb[:],Tb[:],numbatch=10)
Lstar = adknn.L
fLC = lambda y: Lstar
        
dfC = pd.DataFrame(Lstar)
dcate_cobyla, t_hat_cobyla = adknn.CATE(Xb,Xb,Yb,Tb,fLC=fLC,fLT=fLC)
