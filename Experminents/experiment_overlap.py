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
sns.set()

n = 500
p = 2

np.random.seed(0)
diff_mean = []
overlaps = np.sqrt([100,5,1,0.001])
err_array = []
fig, axes = plt.subplots(nrows=1, ncols=4,sharey=True,sharex=True)
for i in range(len(overlaps)):
    overlap = overlaps[i]
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=1, overlap=overlap)
    axi = axes[i]
    sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5,ax=axi)
    df_data_C = df_data.loc[df_data['T']==0][['X0','X1']]
    df_data_T = df_data.loc[df_data['T']==1][['X0','X1']]
    std_diff_mean = np.sqrt(np.matmul(np.matmul((df_data_T.mean(axis=0) - df_data_C.mean(axis=0)).T,np.linalg.inv(df_data[['X0','X1']].cov())),(df_data_T.mean(axis=0) - df_data_C.mean(axis=0))))
    diff_mean.append(std_diff_mean)
    axi.title.set_text('Std. Diff.of Mean = %.2f'%(std_diff_mean))
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10, n_splits=2,n_repeats=3)
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()
    cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())
    err_array.append(cate_df['Relative Error (%)'].to_numpy())
handles, labels = axi.get_legend_handles_labels()
plt.tight_layout()
fig.savefig('Figures/overlap_space.png')

err_array = np.array(err_array).reshape(1,-1)[0]*100
label = []
for i in range(len(overlaps)):
    label += [ '%.2f'%(diff_mean[i]) for j in range(n)]
df = pd.DataFrame()
df['Relative Error (%)'] = err_array
df['Standardized Difference of Means'] = label

fig = plt.figure()
sns.boxenplot(x='Standardized Difference of Means',y='Relative Error (%)',data=df)
plt.yscale('log')
plt.tight_layout()
fig.savefig('Figures/overlap.png')


print(diff_mean)