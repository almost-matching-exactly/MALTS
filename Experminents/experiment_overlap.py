# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:06:34 2020

@author: Harsh
"""

import pandas as pd
import datagen as dg
import numpy as np
import pymalts
import seaborn as sns
import matplotlib.pyplot as plt

n = 200
p = 2

diff_mean = []
overlaps = np.sqrt([100,50,10,0.5,0.05,0.005])
err_array = []
for overlap in overlaps:
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=1, overlap=overlap)
    df_data_C = df_data.loc[df_data['T']==0][['X0','X1']]
    df_data_T = df_data.loc[df_data['T']==1][['X0','X1']]
    diff_mean.append(np.linalg.norm(df_data_T.mean(axis=0) - df_data_C.mean(axis=0)))
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10, n_splits=2,n_repeats=1)
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()
    cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())
    err_array.append(cate_df['Relative Error (%)'].to_numpy())

for i in range(len(overlaps)):
    plt.violinplot(err_array[i],positions=[i])

plt.xticks(ticks=[i for i in range(len(overlaps))],labels=list(map(lambda x:'%.2f'%(x),diff_mean)),rotation=65)
    
plt.xlabel('Norm-2 Diff of Means')
plt.ylabel('Relative Error')
