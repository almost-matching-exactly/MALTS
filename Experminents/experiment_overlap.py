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

ns = [500,2000,4000]
num_cov_dense = 4
num_covs_unimportant = 0
p = num_cov_dense+num_covs_unimportant

np.random.seed(0)
rn_seeds = np.arange(0,50) #[0,1,2,3,4,5,6,7,8,9]
overlaps = [10e-4,10e-2,10e-1,10e0,10e1]

df_err_array = {}
df_err = pd.DataFrame()

sns.set(font_scale=1.5)
# fig,ax = plt.subplots(nrows=len(ns),figsize=(40,100),sharex=True,sharey=True)
for ni in range(len(ns)):
    diff_mean = []
    for i in range(len(overlaps)):
        for rn_seed in rn_seeds:
            # print((ni,i,rn_seed))
            np.random.seed(rn_seed)
            n = ns[ni]
            overlap = overlaps[i]
            df_data, df_true, discrete = dg.data_generation_dense_mixed_endo(n, p, 0, p, 0, rho=0, scale=1, overlap=overlap)
            # axi = axes[i]
            
            # fig = plt.figure()
            # sns.scatterplot(x='X0',y='X1',hue='T',data=df_data,alpha=0.5)
            df_data_C = df_data.loc[df_data['T']==0][['X%d'%(j) for j in range(p)]]
            df_data_T = df_data.loc[df_data['T']==1][['X%d'%(j) for j in range(p)]]
            std_diff_mean = np.sqrt(np.matmul(np.matmul((df_data_T.mean(axis=0) - df_data_C.mean(axis=0)).T,np.linalg.inv(df_data[['X%d'%(j) for j in range(p)]].cov())),(df_data_T.mean(axis=0) - df_data_C.mean(axis=0))))
            print((ni,i,rn_seed,overlap,std_diff_mean))
            # axi.title.set_text('Std. Diff.of Mean = %.2f'%(std_diff_mean))
            
            t_true = df_true['TE']
            ate_true = np.mean(t_true)
        
            err_malts, err_bart, err_crf, err_genmatch, err_psnn, err_full, err_prog = [], [], [], [], [], [], []
            label_malts, label_bart, label_crf, label_genmatch, label_psnn, label_full, label_prog = [], [], [], [], [], [], []
            
            
            m = pymalts.malts_mf( 'Y', 'T', data = df_data, n_splits=5, C=5, k_tr=10, k_est=10 )
            cate_df = m.CATE_df
            cate_df['true.CATE'] = df_true['TE'].to_numpy()
            
            err_malts_mf = [np.nanmean(list(np.array(list( np.abs(t_true - cate_df['avg.CATE']) ))/ate_true ))]
            label_malts = [ 'MALTS' for i in range(len(err_malts_mf)) ]
            err_malts += err_malts_mf
        
        
            #----------------------------------------------------------------------------------------------
            ##Prognostic
            prog_cate = prognostic.prognostic_cv('Y','T',df_data)
            
            err_prog = [np.nanmean(list(np.array(list( np.abs(t_true - prog_cate['avg.CATE']) ))/ate_true ))]
            label_prog = [ 'Prognostic Score' for i in range(len(err_prog)) ]
        
            #----------------------------------------------------------------------------------------------
            ##DBARTS
            bart_cate = bart.bart('Y','T',df_data,n_splits=5)
            
            err_bart = [np.nanmean(list( np.abs(bart_cate['avg.CATE'] - t_true)/ate_true ))]
            label_bart = [ 'BART' for i in range(len(err_bart)) ]
        
            #----------------------------------------------------------------------------------------------
            ##Causal Forest
            crf_cate = causalforest.causalforest('Y','T',df_data,n_splits=5)
            
            err_crf = [np.nanmean(list( np.abs(crf_cate['avg.CATE'] - t_true)/ate_true ))]
            label_crf = [ 'Causal Forest' for i in range(len(err_crf)) ]
        
        
        
            # #---------------------------------------------------------------------------------------------
            ##MATCHIT
            # ate_genmatch, t_hat_genmatch = matchit.matchit('Y','T',df_data,method='genetic')
            ate_psnn, t_hat_psnn = matchit.matchit('Y','T',df_data,method='nearest')
            # ate_full, t_hat_full = matchit.matchit('Y','T',df_data,method='full')
        
        
            # err_genmatch = list( np.abs(t_hat_genmatch['CATE'] - t_true)/ate_true )
            # label_genmatch = [ 'GenMatch' for i in range(len(err_genmatch)) ]
        
        
            err_psnn = [np.nanmean(list( np.abs(t_hat_psnn['CATE'] - t_true)/ate_true ))]
            label_psnn = [ 'Propensity Score' for i in range(len(err_psnn)) ]
        
        
            # #---------------------------------------------------------------------------------------------
            
            err = pd.DataFrame()
            err['Error'] = np.array(err_malts + err_bart + err_crf + err_genmatch + err_psnn + err_full + err_prog)*100
            err['Method'] = label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog
            err['Overlap ($\\epsilon_t$)'] = [overlap for a in range(len(label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog))]
            err['Overlap (Standardized Difference of Means)'] = [std_diff_mean for a in range(len(label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog))]
            err['Overlap'] = ['%.3f\n ($\\epsilon_t$ = %.2f)'%(std_diff_mean,overlap) for a in range(len(label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog))]
            err['n'] = [n for a in range(len(label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog))]
            df_err = df_err.append(err,ignore_index=True)
            
    
    # df_err_n = df_err.loc[df_err['n']==n]
    
    # axi = ax[ni]
    # axi.annotate(s='n=%d'%(ns[ni]),xy=(0, 0.5), xytext=(-axi.yaxis.labelpad-5,0),                    
    #             xycoords=axi.yaxis.label, textcoords='offset points',
    #             size='large', ha='right', va='center',rotation=90)
    # # fig, ax = plt.subplots(figsize=(40,50))
    # if ni==0:
    #     sns.lineplot(hue='Method',y='Error',x='Overlap ($\\epsilon_t$)', data=df_err,style='Method',markers=True,linewidth=3.5,markersize=10, ax=axi)
    #     axi.legend(bbox_to_anchor=(1., 1.05))
    # else:
    #     sns.lineplot(hue='Method',y='Error',x='Overlap ($\\epsilon_t$)', data=df_err,style='Method',markers=True,linewidth=3.5,markersize=10, ax=axi,legend=False)

    # # sns.scatterplot(hue='Method',y='Mean Relative CATE Error (percentage)',x='#Units/#Covariates', data=df_err, legend=False)
    # axi.set_xscale('linear')
    # axi.set_yscale('linear')
    # axi.yaxis.set_major_formatter(ticker.PercentFormatter())

    # sns.boxenplot(hue='Method',y='Relative CATE Error (percentage)',x='Overlap (Standardized Difference of Means)', data=df_err)
    # plt.xticks(rotation=0, horizontalalignment='right')
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    # plt.tight_layout()
    # fig.savefig('Figures/boxplot_multifold_malts_overlap.png')
    

     
g = sns.FacetGrid(data=df_err,row='n',hue='Method',aspect=5,margin_titles=False,legend_out=True)
g.map(sns.lineplot,'Overlap ($\\epsilon_t$)','Error')
plt.xscale('log')
g.savefig('Figures/trend_multifold_malts_overlap.png')
sns.lmplot(x='Overlap (Standardized Difference of Means)', y='Error', data=df_err, hue='Method', row='n',order=2,margin_titles=True,legend_out=False)

df_err.to_csv('Logs/CATE_Multifold_Est_Error_File_o.csv')





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
