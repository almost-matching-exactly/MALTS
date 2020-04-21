# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:21 2020

@author: Harsh
"""
import pandas as pd
import numpy as np

import pymalts

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster

import matchit
import bart
import causalforest
import prognostic

import warnings
warnings.filterwarnings("ignore")


nsw = pd.read_stata('http://www.nber.org/~rdehejia/data/nsw.dta')
psid_control = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls.dta')
psid_control2 = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls2.dta')
psid_control3 = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls3.dta')

nsw = nsw.drop(columns=['data_id','re74'],errors='ignore')
psid_control = psid_control.drop(columns=['data_id','re74'],errors='ignore')
psid_control2 = psid_control2.drop(columns=['data_id','re74'],errors='ignore')
psid_control3 = psid_control3.drop(columns=['data_id','re74'],errors='ignore')

data = nsw.append(psid_control2,ignore_index=True)

#Matching on NSW Male Subset of Lalonde's Data
np.random.seed(0)
m_nsw = pymalts.malts_mf('re78', 'treat', data=nsw,
                     discrete=['black','hispanic','married','nodegree'],
                     k_est=150,n_splits=5)

cate_df_nsw = m_nsw.CATE_df['CATE']
cate_df_nsw['avg.CATE'] = cate_df_nsw.mean(axis=1)
cate_df_nsw['std.CATE'] = cate_df_nsw.std(axis=1)
cate_df_nsw['re78'] = m_nsw.CATE_df['outcome'].mean(axis=1)
cate_df_nsw['treat'] = m_nsw.CATE_df['treatment'].mean(axis=1)
print(np.mean(cate_df_nsw['avg.CATE']))
m_opt_list = pd.concat(m_nsw.M_opt_list)
ate_df_nsw = np.mean(cate_df_nsw,axis=0)
ate_df_nsw = ate_df_nsw.rename({'CATE':'ATE','avg.CATE':'avg.ATE'}).drop(['std.CATE'])
ate_df_nsw['std.ATE'] = ate_df_nsw['ATE'].std()

e_bias = (np.mean(cate_df_nsw['avg.CATE']) - 886)*100/886
df_nsw_result = pd.DataFrame([['MALTS',np.mean(cate_df_nsw['avg.CATE']),e_bias]],
                             columns=['Method','ATE Estimate','Estimation Bias (%)'])


#Matching on NSW+PSID Male Subset of Lalonde's Data
np.random.seed(0)
m = pymalts.malts_mf('re78', 'treat', data=data,
                     discrete=['black','hispanic','married','nodegree'],
                     k_est=150,n_splits=5)

df_full = m.CATE_df
cate_df = m.CATE_df['CATE']
cate_df['avg.CATE'] = cate_df.mean(axis=1)
cate_df['avg.Diam'] = df_full['diameter'].mean(axis=1)
cate_df['std.CATE'] = cate_df.std(axis=1)
cate_df['re78'] = m.CATE_df['outcome'].mean(axis=1)
cate_df['treat'] = m.CATE_df['treatment'].mean(axis=1)

np.random.seed(0)
clust = cluster.KMeans(n_clusters=3).fit(cate_df['avg.Diam'].to_numpy().reshape(-1,1))

fig = plt.figure()
sns.scatterplot(y=cate_df['avg.CATE'],x=cate_df['avg.Diam'],hue=clust.labels_,alpha=0.6,palette='Set1')
plt.axvline(8.5e7)
plt.xscale('log')
plt.tight_layout()
fig.savefig('lalonde_pruning.png')

print(np.mean(cate_df['avg.CATE']))
print(cate_df.loc[cate_df['avg.Diam']<8.5e7]['avg.CATE'].mean())
print(cate_df.loc[cate_df['treat']==1]['avg.CATE'].mean())

m_opt_list = pd.concat(m.M_opt_list)
ate_df = np.mean(cate_df,axis=0)
ate_df = ate_df.rename({'CATE':'ATE','avg.CATE':'avg.ATE'}).drop(['std.CATE'])
ate_df['avg.ATE-Pruned'] = cate_df.loc[cate_df['avg.Diam']<8.5e7]['avg.CATE'].mean()
ate_df['std.ATE'] = ate_df['ATE'].std()

e_bias = [ (ate_df['avg.ATE'] - 886)*100/886 , (ate_df['avg.ATE-Pruned'] - 886)*100/886]
df_result = pd.DataFrame([[ 'MALTS', ate_df['avg.ATE'], e_bias[0] ],
                              [ 'MALTS-Pruned', ate_df['avg.ATE-Pruned'], e_bias[1] ]],
                             columns=['Method','ATE Estimate','Estimation Bias (%)'])


#other methods

##Genmatch
np.random.seed(0)
ate_nsw_psnn,_ = matchit.matchit('re78','treat',data=nsw,method='genetic')
ate_psnn,_ = matchit.matchit('re78','treat',data=data,method='genetic')

ate = np.array([ate_nsw_psnn,ate_psnn])
e_bias = (ate - 886)*100/886

df_nsw_result = df_nsw_result.append(pd.DataFrame( [['GenMatch',ate[0],e_bias[0]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))

df_result = df_result.append(pd.DataFrame( [['GenMatch',ate[1],e_bias[1]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))



##PSNN
np.random.seed(0)
ate_nsw_psnn,_ = matchit.matchit('re78','treat',data=nsw,method='nearest',replace=True)
ate_psnn,_ = matchit.matchit('re78','treat',data=data,method='nearest',replace=True)

ate = np.array([ate_nsw_psnn,ate_psnn])
e_bias = (ate - 886)*100/886

df_nsw_result = df_nsw_result.append(pd.DataFrame( [['Propensity Score',ate[0],e_bias[0]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))

df_result = df_result.append(pd.DataFrame( [['Propensity Score',ate[1],e_bias[1]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))


##Prognostic
np.random.seed(0)
cate_est_nsw = prognostic.prognostic_cv('re78','treat',data=nsw)
cate_est = prognostic.prognostic_cv('re78','treat',data=data)

ate_est_nsw = cate_est_nsw.mean(axis=0).mean()
ate_est = cate_est.mean(axis=0).mean()

ate = np.array([ate_est_nsw,ate_est])
e_bias = (ate - 886)*100/886

df_nsw_result = df_nsw_result.append(pd.DataFrame( [['Prognostic Score',ate[0],e_bias[0]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))

df_result = df_result.append(pd.DataFrame( [['Prognostic Score',ate[1],e_bias[1]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))


##BART
np.random.seed(0)
cate_est_nsw = bart.bart('re78','treat',nsw,5)
cate_est = bart.bart('re78','treat',data,5)
ate_est_nsw = cate_est_nsw.mean(axis=0).mean()
ate_est = cate_est.mean(axis=0).mean()

ate = np.array([ate_est_nsw,ate_est])
e_bias = (ate - 886)*100/886

df_nsw_result = df_nsw_result.append(pd.DataFrame( [['BART-CV',ate[0],e_bias[0]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))

df_result = df_result.append(pd.DataFrame( [['BART-CV',ate[1],e_bias[1]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))


##Causal Forest
np.random.seed(0)
cate_est_nsw = causalforest.causalforest('re78','treat',nsw,5)
cate_est = causalforest.causalforest('re78','treat',data,5)
ate_est_nsw = cate_est_nsw.mean(axis=0).mean()
ate_est = cate_est.mean(axis=0).mean()

ate = np.array([ate_est_nsw,ate_est])
e_bias = (ate - 886)*100/886

df_nsw_result = df_nsw_result.append(pd.DataFrame( [['Causal Forest-CV',ate[0],e_bias[0]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))

df_result = df_result.append(pd.DataFrame( [['Causal Forest-CV',ate[1],e_bias[1]]],
                                          columns=['Method','ATE Estimate','Estimation Bias (%)']))


df_nsw_result = df_nsw_result.set_index('Method')
df_result = df_result.set_index('Method')

df_result = df_result.drop(index=['MALTS'])
df_result = df_result.rename({'MALTS-Pruned':'MALTS'})

df_nsw_result.to_latex('lalonde_NSW.tex')
df_result.to_latex('lalonde_PSID2.tex')