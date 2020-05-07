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

n = 500
p = 2

np.random.seed(0)
diff_mean = []
overlaps = np.sqrt([100,5,0.1,0.001])
err_array = []
fig, axes = plt.subplots(nrows=2, ncols=2,sharey=True,sharex=True)
for i in range(len(overlaps)):
    overlap = overlaps[i]
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=1, overlap=overlap)
    sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5,ax=axes[i//2, i%2])
    df_data_C = df_data.loc[df_data['T']==0][['X0','X1']]
    df_data_T = df_data.loc[df_data['T']==1][['X0','X1']]
    diff_mean.append(np.linalg.norm(df_data_T.mean(axis=0) - df_data_C.mean(axis=0))/np.linalg.norm(df_data[['X0','X1']].mean(axis=0)))
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10, n_splits=2,n_repeats=2)
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()
    cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())
    err_array.append(cate_df['Relative Error (%)'].to_numpy())

err_array = np.array(err_array).reshape(1,-1)[0]*100
label = []
for i in range(len(overlaps)):
    label += [ '%.2f'%(diff_mean[i]) for j in range(n)]
df = pd.DataFrame()
df['Relative Error (%)'] = err_array
df['Standardized Difference of Means'] = label

fig = plt.figure()
sns.boxenplot(x='Standardized Difference of Means',y='Relative Error (%)',data=df)
fig.savefig('overlap.png')
# for i in range(len(overlaps)):
#     plt.violinplot(err_array[i]*100,positions=[diff_mean[i]])

# plt.xticks(ticks=[i for i in range(len(overlaps))],labels=list(map(lambda x:'%.2f'%(x),diff_mean)),rotation=65)
    
plt.xlabel('Standardize difference of means')
plt.ylabel('Relative Error (%)')

print(diff_mean)