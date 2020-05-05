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

n = 1000
p = 2

std = np.sqrt([1,2,3,4,5])
coverage_malts_std = []
coverage_true_std = []
for scale in std:
    for rep in range(20):
        df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=scale, coverage=True)
        m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=10, k_est=20, n_splits=4, n_repeats=3)
        cate_df = m.CATE_df
        cate_df['true.CATE'] = df_true['TE'].to_numpy()
        cate_df['covered.MALTS.var %.1f'%(scale**2)] = np.abs(cate_df['avg.CATE'].iloc[0:9] - cate_df['true.CATE'].iloc[0:9]) < 2*cate_df['std.CATE'].iloc[0:9]
        cate_df['covered.MALTS.gbr.var %.1f'%(scale**2)] = np.abs(cate_df['avg.gbr.CATE'].iloc[0:9] - cate_df['true.CATE'].iloc[0:9]) < 2*cate_df['std.gbr.CATE'].iloc[0:9]
        # cate_df['covered.True.std'] = np.abs(cate_df['avg.CATE'] - cate_df['true.CATE']) < 4*scale
        coverage_df = pd.DataFrame(cate_df[['covered.MALTS.var %.1f'%(scale**2),'covered.MALTS.gbr.var %.1f'%(scale**2)]].iloc[0:9])               
        print(coverage_df)
        coverage_malts_std.append(coverage_df)
        # coverage_true_std.append(cate_df['covered.True.std'].mean())
df_coverage = pd.concat(coverage_malts_std,axis=1)
df_coverage_T = df_coverage.transpose()

fig, ax = plt.subplots()
sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5)
sns.scatterplot(x='X0',y='X1',marker="*",s=50*cate_df['std.gbr.CATE'].iloc[:9],hue=df_coverage_T.mean(axis=0),data=df_data.iloc[0:9])
df_coverage_T = df_coverage_T.astype(int)
group = df_coverage_T.groupby(level=0)
group.mean().to_csv('coverage_1000.csv')

# df_coverage = pd.DataFrame()
# df_coverage['True Variance'] = np.square(std)
# df_coverage['Coverage w/ true std'] = coverage_true_std
# df_coverage['Coverage w/ MALTS std'] = coverage_malts_std

# sns.set()
# plt.plot(df_coverage['True Variance'],df_coverage['Coverage w/ true std'])
# plt.plot(df_coverage['True Variance'],df_coverage['Coverage w/ MALTS std'])
# plt.ylabel('Coverage')
# plt.xlabel('True Variance')
# plt.legend(['Coverage w/ True std','Coverage w/ MALTS std'])
# plt.ylim((0.80,1.005))
