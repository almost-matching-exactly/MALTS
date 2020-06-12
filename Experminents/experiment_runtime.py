# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:05:00 2020

@author: Harsh
"""


import pandas as pd
import datagen as dg
import numpy as np
import pymalts
import seaborn as sns
import matplotlib.pyplot as plt
import time
sns.set()



ns = np.array([100,200,500,1000,2000,5000,10000])
ps = [2,10,20,50,100,500,1000]
times = []
error_rate = []
for n in ns:
    np.random.seed(0)
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, ps[0], 0, 0, 0)
    t1 = time.time()
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10)
    t2 = time.time()
    
    err = (m.CATE_df['avg.CATE'] - df_true['TE']).mean()/df_true['TE'].mean()
    times.append((t2-t1))
    error_rate.append(err)
    
fig,ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(ns[:len(times)],times)
ax[1].plot(ns[:len(times)],np.abs(error_rate))
