# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:31:42 2020

@author: Harsh
"""


import numpy as np
import pandas as pd

import pymalts
import prognostic
import matchit
import bart
import causalforest

import datagen as dg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
np.random.seed(0)

num_samples = 2500
imp_c = 15
imp_d = 0
unimp_c = 25
unimp_d = 0

df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(num_samples, imp_c, imp_d, unimp_c, unimp_d, rho=0, overlap=1)

m = pymalts.malts_mf( 'Y', 'T', data = df_data, discrete=discrete, k_tr=15, k_est=80, n_splits=5 )
cate_df = m.CATE_df

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
fig.savefig('Figures/trueVSestimatedCATE_malts_continuous.png')

##OTHERS METHODS
ate_psnn, t_psnn = matchit.matchit('Y','T',data=df_data,method='nearest',replace=True)

df_err_psnn = pd.DataFrame()
df_err_psnn['Method'] = ['Propensity Score' for i in range(t_psnn.shape[0])] 
df_err_psnn['Relative Error (%)'] = np.abs((t_psnn['CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())


ate_gen, t_gen = matchit.matchit('Y','T',data=df_data,method='genetic',replace=True)

df_err_gen = pd.DataFrame()
df_err_gen['Method'] = ['GenMatch' for i in range(t_gen.shape[0])] 
df_err_gen['Relative Error (%)'] = np.abs((t_gen['CATE'].to_numpy() - df_true['TE'].to_numpy())/df_true['TE'].mean())


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

df_err['Relative Error (%)'] = df_err['Relative Error (%)'] * 100

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=3)

fig, ax = plt.subplots(figsize=(40,50))
sns.boxenplot(x='Method',y='Relative Error (%)',data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/boxplot_multifold_malts_continuous.png')
import textwrap
ax.set_xticklabels([textwrap.fill(t.get_text(), 10)  for t in ax.get_xticklabels()])
 
fig, ax = plt.subplots(figsize=(40,50))
sns.violinplot(x='Method',y='Relative Error (%)',data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/violin_multifold_malts_continuous.png')

df_err.to_csv('df_err_continuous.csv')

'''
##VISUALIZING Matched Group Matrix
fig = plt.figure()
sns.heatmap(m.MG_matrix)
fig.savefig('Figures/heatmap_malts_matched_group_continuous.png')

import networkx as nx
import matplotlib as mpl
m1 = pymalts.malts_mf( 'Y', 'T', data = df_data.head(100), discrete=discrete, k_tr=2, k_est=2, n_splits=5, estimator='mean' )

np.fill_diagonal(m1.MG_matrix.to_numpy(),0)
fig = plt.figure()
sns.heatmap(m1.MG_matrix.to_numpy())
fig.savefig('Figures/demo_heatmap_malts_matched_group_continuous.png')

G = nx.from_numpy_matrix(m1.MG_matrix.to_numpy(),create_using=nx.DiGraph())
pos = nx.layout.fruchterman_reingold_layout(G)
pos2 = nx.layout.spectral_layout(G)

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

node_sizes = [20 for i in range(len(G))]
M = G.number_of_edges()

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                               node_color=df_data.head(100)['T'], label=df_data.head(100)['T'],
                               cmap=plt.cm.Set1)

edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowsize=10, edge_color=weights, alpha=0.2,
                               edge_cmap=plt.cm.Blues, width=weights)

plt.colorbar(nodes)

'''