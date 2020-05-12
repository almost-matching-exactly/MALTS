# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:35:34 2020

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
diff_mean = []
bias = [0,2,10,25]
err_array = []
fig, axes = plt.subplots(nrows=2, ncols=4)
for i in range(len(bias)):
    b = bias[i]
    np.random.seed(0)
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=1, overlap=20, bias=b)
    axi = axes[i//2, 2*(i%2)]
    sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5,ax=axi)
    
    print('#Treated / #Control: %.2f'%(df_data.loc[df_data['T']==1].shape[0]/df_data.loc[df_data['T']==0].shape[0]))
    axi.title.set_text('#Treated / #Control: %.2f'%(df_data.loc[df_data['T']==1].shape[0]/df_data.loc[df_data['T']==0].shape[0]))
    
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10, n_splits=2,n_repeats=2)
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()
    cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())
    
    m_re = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=5, k_est=10, n_splits=2,n_repeats=2,reweight=True)
    cate_df_re = m_re.CATE_df
    cate_df_re['true.CATE'] = df_true['TE'].to_numpy()
    cate_df_re['Relative Error (%)'] = np.abs((cate_df_re['avg.CATE']-cate_df_re['true.CATE'])/cate_df_re['true.CATE'].mean())
    
    df_err_malts = pd.DataFrame()
    df_err_malts['Method'] = ['MALTS' for i in range(cate_df.shape[0])] + ['MALTS (Reweighted)' for i in range(cate_df.shape[0])]
    df_err_malts['Relative Error (%)'] = cate_df['Relative Error (%)'].to_list() + cate_df_re['Relative Error (%)'].to_list()
    
    axi = axes[i//2, 2*(i%2) + 1]
    sns.boxenplot(x='Method',y='Relative Error (%)',data=df_err_malts,ax=axi)