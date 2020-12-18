# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:29:30 2020

@author: Harsh
"""


import pymalts
import prognostic
import matchit
import bart
import causalforest

import sklearn.datasets as dg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import scipy

def u(x):
    T = []
    for row in x:
        l = scipy.special.expit(row[0]+row[1]-0.5+np.random.normal(0,1))
        t = int( l > 0.5 )
        T.append(t)
    return np.array(T)

def TE(X):
    return X[:,2]*np.cos(np.pi * X[:, 0] * X[:, 1])

np.random.seed(0)

##GENERATE DATA
n = 6000
p = 10
e = 1
X,Y0 = dg.make_friedman1(n_samples=n, n_features=p, noise=e, random_state=0)
Y1 = Y0 + TE(X)
T = u(X)
Y = T*Y1 + (1-T)*Y0

columns = ['X%d'%(i) for i in range(p)] + ['Y','T']
df_data = pd.DataFrame(np.hstack((X,Y.reshape(-1,1),T.reshape(-1,1))),columns=columns)
df_true = pd.DataFrame(np.hstack((Y0.reshape(-1,1),Y1.reshape(-1,1),TE(X).reshape(-1,1),T.reshape(-1,1))),columns=['Y0','Y1','TE','T'])

##MALTS
m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=[], k_tr=20, k_est=100, n_splits=5, n_repeats=2)
cate_df = m.CATE_df

##MALTS Result Analysis
cate_df['true.CATE'] = df_true['TE'].to_numpy()

cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())
cate_df['Method'] = ['MALTS' for i in range(cate_df.shape[0])]

df_err_malts = pd.DataFrame()
df_err_malts['Method'] = ['MALTS' for i in range(cate_df.shape[0])]
df_err_malts['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/cate_df['true.CATE'].mean())

##PLOTTING MALTS RESULTS
fig, ax = plt.subplots()
sns.scatterplot(x='true.CATE',y='avg.CATE',size='std.CATE',hue='T',alpha=0.2,sizes=(10,200),data=cate_df)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.xlabel('True CATE')
plt.ylabel('Estimated CATE')
fig.savefig('Figures/trueVSestimatedCATE_malts_friedman.png')

##OTHERS METHODS
ate_psnn, t_psnn = matchit.matchit('Y','T',data=df_data,method='nearest',replace=True)

df_err_psnn = pd.DataFrame()
df_err_psnn['Method'] = ['Propensity Score' for i in range(t_psnn.shape[0])] 
df_err_psnn['Relative Error (%)'] = np.abs((t_psnn['CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())

'''
ate_gen, t_gen = matchit.matchit('Y','T',data=df_data,method='genetic',replace=True)
'''
df_err_gen = pd.DataFrame()

df_err_gen['Method'] = []#['GenMatch' for i in range(t_gen.shape[0])] 
df_err_gen['Relative Error (%)'] = []#np.abs((t_gen['CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())


cate_est_prog = prognostic.prognostic_cv('Y', 'T', df_data,n_splits=5)

df_err_prog = pd.DataFrame()
df_err_prog['Method'] = ['Prognostic Score' for i in range(cate_est_prog.shape[0])] 
df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())


cate_est_bart = bart.bart('Y','T',df_data,n_splits=5)

df_err_bart = pd.DataFrame()
df_err_bart['Method'] = ['BART' for i in range(cate_est_bart.shape[0])] 
df_err_bart['Relative Error (%)'] = np.abs((cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())


cate_est_cf = causalforest.causalforest('Y','T',df_data,n_splits=5)

df_err_cf = pd.DataFrame()
df_err_cf['Method'] = ['Causal Forest' for i in range(cate_est_cf.shape[0])] 
df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())

df_err = pd.DataFrame(columns = ['Method','Relative Error (%)'])
df_err = df_err.append(df_err_malts).append(df_err_psnn).append(df_err_gen).append(df_err_prog).append(df_err_bart).append(df_err_cf)

df_err_2 = pd.DataFrame(columns = ['Method','Relative Error (%)'])
df_err_2 = df_err_2.append(df_err_malts).append(df_err_bart).append(df_err_cf)


df_err['Relative Error (%)'] = df_err['Relative Error (%)'] * 100
df_err_2['Relative Error (%)'] = df_err_2['Relative Error (%)'] * 100

sns.set_context("paper")
sns.set_style("darkgrid")

fig, ax = plt.subplots()
sns.boxenplot(x='Method',y='Relative Error (%)',data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/boxplot_multifold_malts_friedman.png')
 
fig, ax = plt.subplots()
sns.violinplot(x='Method',y='Relative Error (%)',data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/violin_multifold_malts_friedman.png')

fig, ax = plt.subplots()
sns.boxenplot(x='Method',y='Relative Error (%)',data=df_err_2)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/boxplot_multifold_malts_friedman_2.png')
 
fig, ax = plt.subplots()
sns.violinplot(x='Method',y='Relative Error (%)',data=df_err_2)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/violin_multifold_malts_friedman_2.png')


df_err.to_csv('df_err_friedman.csv')
