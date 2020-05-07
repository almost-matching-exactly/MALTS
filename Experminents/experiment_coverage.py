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
import tabulate
n = 250
p = 2

std = np.sqrt([1,2,3,4])
coverage_malts_std = pd.DataFrame()
coverage_true_std = []
for scale in std:
    for rep in range(30):
        df_data, df_true, discrete, dense_coef = dg.data_generation_dense_mixed_endo(n, p, 0, 0, 0, rho=0, scale=scale, coverage=True)
        m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=10, k_est=10, n_splits=2, n_repeats=2)
        cate_df = m.CATE_df
        cate_df['true.CATE'] = df_true['TE'].to_numpy()
        cate_df['covered.MALTS.var %.1f'%(scale**2)] = np.abs(cate_df['avg.CATE'].iloc[0:9] - cate_df['true.CATE'].iloc[0:9]) < 2*cate_df['std.CATE'].iloc[0:9]
        cate_df['covered.MALTS.gbr.var %.1f'%(scale**2)] = np.abs(cate_df['avg.gbr.CATE'].iloc[0:9] - cate_df['true.CATE'].iloc[0:9]) < 2*cate_df['std.gbr.CATE'].iloc[0:9]
        # cate_df['covered.True.std'] = np.abs(cate_df['avg.CATE'] - cate_df['true.CATE']) < 4*scale
        
        coverage_df = pd.DataFrame()
        coverage_df['var'] = [(scale**2) for i in range(0,9)]
        coverage_df['Sample Variance'] = cate_df['covered.MALTS.var %.1f'%(scale**2)].iloc[0:9].astype(int)
        coverage_df['Gradient Boosting Regressor Variance'] = cate_df['covered.MALTS.gbr.var %.1f'%(scale**2)].iloc[0:9].astype(int)
        
        # coverage_df = pd.DataFrame(cate_df[['covered.MALTS.var %.1f'%(scale**2),'covered.MALTS.gbr.var %.1f'%(scale**2)]].iloc[0:9])               
        # print(tabulate.tabulate(cate_df.iloc[:9][['avg.CATE','avg.gbr.CATE','true.CATE','std.CATE','std.gbr.CATE','covered.MALTS.var %.1f'%(scale**2),'covered.MALTS.gbr.var %.1f'%(scale**2)]], headers='keys'))
        # print(df_data[['X0','X1']].iloc[9:19])
        # print(dense_coef)
        coverage_malts_std = coverage_malts_std.append(coverage_df)
        # coverage_true_std.append(cate_df['covered.True.std'].mean())
df_coverage = coverage_malts_std
df_coverage.to_csv('coverage.csv')
fig, axes = plt.subplots(nrows=3, ncols=3,sharey=True,sharex=True)
for i in range(0,9):
    df_coverage_i = df_coverage.loc[i]
    df_coverage_i['Sample Variance'] = df_coverage_i['Sample Variance'].astype(int)
    df_coverage_i['Gradient Boosting Regressor Variance'] = df_coverage_i['Gradient Boosting Regressor Variance'].astype(int)
    df_coverage_i = df_coverage_i.groupby('var').mean()
    axi = axes[i//3,i%3]
    df_coverage_i.plot(ax=axi,legend=False,grid=True)
    axi.title.set_text(df_data[['X0','X1']].iloc[i].to_string())
    # axi.axhline(y=0.9,c='black',alpha=0.2)
    # axi.axhline(y=0.8,c='black',alpha=0.2)
handles, labels = axi.get_legend_handles_labels()
fig.legend(handles, ['Boosting Variance','Gradient Boosting Regressor Variance'], loc='lower right',ncol=2)
plt.tight_layout()
plt.yticks(ticks=[1,0.8,0.9,0.5,0],label=['1','0.8','0.9','0.5','0'])

fig, axes = plt.subplots(nrows=2, ncols=2,sharey=True,sharex=True)
for i in range(1,5):
    df_coverage_i = df_coverage.loc[i]
    df_coverage_i['Sample Variance'] = df_coverage_i['Sample Variance'].astype(int)
    df_coverage_i['Gradient Boosting Regressor Variance'] = df_coverage_i['Gradient Boosting Regressor Variance'].astype(int)
    df_coverage_i = df_coverage_i.groupby('var').mean()
    axi = axes[(i-1)//2,(i-1)%2]
    df_coverage_i.plot(ax=axi,legend=False,grid=True)
    axi.title.set_text(df_data[['X0','X1']].iloc[i].to_string())
    # axi.axhline(y=0.9,c='black',alpha=0.2)
    # axi.axhline(y=0.8,c='black',alpha=0.2)
handles, labels = axi.get_legend_handles_labels()
fig.legend(handles, ['Boosting Variance','Gradient Boosting Regressor Variance'], loc='lower right',ncol=2)
plt.tight_layout()
plt.yticks(ticks=[1,0.8,0.9,0.5,0],label=['1','0.8','0.9','0.5','0'])


# df_coverage_T = df_coverage.transpose()

fig, ax = plt.subplots()
sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5)
sns.scatterplot(x='X0',y='X1',marker="*",s=200,color='black',data=df_data.iloc[0:9])
# df_coverage_T = df_coverage_T.astype(int)
# group = df_coverage_T.groupby(level=0)
# group.mean().to_csv('coverage_%d_%d.csv'%(n,p))

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
