# -*- coding: utf-8 -*-
"""
Created on Fri May 04 21:31:11 2018

@author: Harsh
"""
import numpy as np
import pandas as pd
import os

def read_covariates():
    f = 'Data/x.csv'
    df = pd.DataFrame.from_csv(f)
    return df

def read_treatment_outcome():
    ls = os.listdir('Data/practice_censoring/censoring')
    ls_t = [s for s in ls if not s.endswith('_cf.csv')]
    ls_cf = [s for s in ls if s.endswith('_cf.csv')]
    readfile = lambda s: pd.DataFrame.from_csv('Data/practice_censoring/censoring/'+s)
    df_t_array = list(map( readfile, ls_t ))
    df_cf_array = list(map( readfile, ls_cf ))
    df_t = pd.concat(df_t_array)
    df_cf = pd.concat(df_cf_array)
    return df_t, df_cf

X = read_covariates()
df_t, df_cf = read_treatment_outcome()
X1 = X.join(df_t)
X1 = X1.join(df_cf)
Xfull = X1[X1.y.notnull()]
Xb,Yb,Tb,Y0b,Y1b = np.array(Xfull[Xfull.columns[0:177]]), np.array(Xfull['y']), np.array(Xfull['z']), np.array(Xfull['y0']),np.array(Xfull['y1'])
