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

std = np.sqrt([1,2,3,4,5])
coverage_malts_std = []
coverage_true_std = []
for scale in std:
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=scale)
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=10, k_est=10, n_splits=2,n_repeats=10)
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()
    cate_df['covered.MALTS.std'] = np.abs(cate_df['avg.CATE'] - cate_df['true.CATE']) < 2*cate_df['std.CATE']
    cate_df['covered.True.std'] = np.abs(cate_df['avg.CATE'] - cate_df['true.CATE']) < 4*scale
    coverage_malts_std.append(cate_df['covered.MALTS.std'].mean())
    coverage_true_std.append(cate_df['covered.True.std'].mean())
    
df_coverage = pd.DataFrame()
df_coverage['True Variance'] = np.square(std)
df_coverage['Coverage w/ true std'] = coverage_true_std
df_coverage['Coverage w/ MALTS std'] = coverage_malts_std

sns.set()
plt.plot(df_coverage['True Variance'],df_coverage['Coverage w/ true std'])
plt.plot(df_coverage['True Variance'],df_coverage['Coverage w/ MALTS std'])
plt.ylabel('Coverage')
plt.xlabel('True Variance')
plt.legend(['Coverage w/ True std','Coverage w/ MALTS std'])
plt.ylim((0.80,1.005))
