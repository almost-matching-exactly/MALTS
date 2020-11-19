#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:50:50 2019

@author: harshparikh
"""
import numpy as np
import pandas as pd
import pymalts
import prognostic
import bart
import causalforest
import matchit
import datagen as dg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
np.random.seed(0)
sns.set()

n = 1024
num_cov_dense = np.array([8,16,32])
p = num_cov_dense

np.random.seed(0)
diff_mean = []

overlap = 20

df_err = pd.DataFrame()
for i in range(len(p)):
    df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p[i], 0, p[i], 0, rho=0, scale=1, overlap=overlap)
    
    df_data_C = df_data.loc[df_data['T']==0][['X%d'%(j) for j in range(p[i])]]
    df_data_T = df_data.loc[df_data['T']==1][['X%d'%(j) for j in range(p[i])]]
    
    std_diff_mean = np.sqrt(np.matmul(np.matmul((df_data_T.mean(axis=0) - df_data_C.mean(axis=0)).T,np.linalg.inv(df_data[['X%d'%(j) for j in range(p[i])]].cov())),(df_data_T.mean(axis=0) - df_data_C.mean(axis=0))))
    diff_mean.append(std_diff_mean)
    print(std_diff_mean)
    
    t_true = df_true['TE']
    ate_true = np.mean(t_true)

    err_malts, err_bart, err_crf, err_genmatch, err_psnn, err_full, err_prog = [], [], [], [], [], [], []
    label_malts, label_bart, label_crf, label_genmatch, label_psnn, label_full, label_prog = [], [], [], [], [], [], []
    
    m = pymalts.malts_mf( 'Y', 'T', data = df_data, n_splits=4, C=5, k_tr=10, k_est=50 )
    cate_df = m.CATE_df
    cate_df['true.CATE'] = df_true['TE'].to_numpy()

    
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
    fig.savefig('Figures/trueVSestimatedCATE_malts_multifold.png')
    
    err_malts_mf = list(np.array(list( np.abs(t_true - cate_df['avg.CATE']) ))/ate_true )
    label_malts = [ 'MALTS' for i in range(len(err_malts_mf)) ]
    err_malts += err_malts_mf

    
    #----------------------------------------------------------------------------------------------
    ##Prognostic
    prog_cate = prognostic.prognostic_cv('Y','T',df_data)
    
    err_prog = list(np.array(list( np.abs(t_true - prog_cate['avg.CATE']) ))/ate_true )
    label_prog = [ 'Prognostic Score' for i in range(len(err_prog)) ]
    
    #----------------------------------------------------------------------------------------------
    ##DBARTS
    bart_cate = bart.bart('Y','T',df_data,n_splits=5)
    
    err_bart = list( np.abs(bart_cate['avg.CATE'] - t_true)/ate_true )
    label_bart = [ 'BART' for i in range(len(err_bart)) ]
    
    
    #----------------------------------------------------------------------------------------------
    ##Causal Forest
    crf_cate = causalforest.causalforest('Y','T',df_data,n_splits=5)
    
    err_crf = list( np.abs(crf_cate['avg.CATE'] - t_true)/ate_true )
    label_crf = [ 'Causal Forest' for i in range(len(err_crf)) ]



    # #---------------------------------------------------------------------------------------------
    ##MATCHIT
    ate_genmatch, t_hat_genmatch = matchit.matchit('Y','T',df_data,method='genetic')
    
    ate_psnn, t_hat_psnn = matchit.matchit('Y','T',df_data,method='nearest')
    

    err_genmatch = list( np.abs(t_hat_genmatch['CATE'] - t_true)/ate_true )
    label_genmatch = [ 'GenMatch' for i in range(len(err_genmatch)) ]
    

    err_psnn = list( np.abs(t_hat_psnn['CATE'] - t_true)/ate_true )
    label_psnn = [ 'Propensity Score' for i in range(len(err_psnn)) ]


    # #---------------------------------------------------------------------------------------------
    
    
    err = pd.DataFrame()
    err['Relative CATE Error (percentage)'] = np.array(err_malts + err_bart + err_crf + err_genmatch + err_psnn + err_full + err_prog)*100
    err['Method'] = label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog
    err['#Covariates/#Units'] = [2*p[i]/n for a in range(len(label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog))]
    
    df_err = df_err.append(err,ignore_index=True)

df_err['#Covariates/#Units'] = df_err['#Covariates/#Units'].round(decimals=7)

sns.set(font_scale=2.,context='paper')
fig, ax = plt.subplots(figsize=(40,50))
sns.boxenplot(hue='Method',y='Relative CATE Error (percentage)',x='#Covariates/#Units', data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.ylim((-50,600))
plt.tight_layout()
fig.savefig('Figures/boxplot_multifold_malts_p.png')
 
fig, ax = plt.subplots(figsize=(40,50))
sns.violinplot(hue='Method',y='Relative CATE Error (percentage)',x='#Covariates/#Units', data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/violin_multifold_malts_p.png')

df_err.to_csv('Logs/CATE_Multifold_Est_Error_File_p.csv')





# handles, labels = axi.get_legend_handles_labels()
# plt.tight_layout()
# fig.savefig('Figures/overlap_space.png')

# err_array = np.array(err_array).reshape(1,-1)[0]*100
# label = []
# for i in range(len(overlaps)):
#     label += [ '%.2f'%(diff_mean[i]) for j in range(n)]
# df = pd.DataFrame()
# df['Relative Error (%)'] = err_array
# df['Standardized Difference of Means'] = label

# fig = plt.figure()
# sns.boxenplot(x='Standardized Difference of Means',y='Relative Error (%)',data=df)
# plt.yscale('log')
# plt.tight_layout()
# fig.savefig('Figures/overlap.png')


# print(diff_mean)
