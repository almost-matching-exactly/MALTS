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
import datagen as dg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
np.random.seed(0)

numExample = 2000
num_cov_dense = 10
num_covs_unimportant = 25
n_est = 2000
num_covariates = num_cov_dense+num_covs_unimportant

df_train, df_true_train = dg.data_generation_dense_endo(numExample, num_cov_dense, num_covs_unimportant,rho=0)

X,Y,T = np.array(df_train[df_train.columns[0:num_covariates]]), np.array(df_train['Y']), np.array(df_train['T'])

df_est, df_true_est = dg.data_generation_dense_endo(n_est, num_cov_dense, num_covs_unimportant,rho=0.2)

df_data = df_train.append(df_est)
df_data_true = df_true_train.append(df_true_est)

Xtest,Ytest,Ttest = np.array(df_est[df_est.columns[0:num_covariates]]), np.array(df_est['Y']), np.array(df_est['T'])
t_true = df_true_est['TE'].to_numpy()
ate_true = np.mean(t_true)
#del Xtest,Ytest,Ttest,df,dense_bs, treatment_eff_coef

err_malts, err_bart, err_crf, err_genmatch, err_psnn, err_full, err_prog = [], [], [], [], [], [], []
label_malts, label_bart, label_crf, label_genmatch, label_psnn, label_full, label_prog = [], [], [], [], [], [], []

m = pymalts.malts_mf( 'Y', 'T', data = df_data, n_splits=5 )
cate_df = m.CATE_df['CATE']
cate_df['avg.CATE'] = cate_df.mean(axis=1)
cate_df['std.CATE'] = cate_df.std(axis=1)
cate_df['outcome'] = m.CATE_df['outcome'].mean(axis=1)
cate_df['treatment'] = m.CATE_df['treatment'].mean(axis=1)
cate_df['true.CATE'] = df_data_true['TE'].to_numpy()
cate_df['err.CATE'] = np.abs(cate_df['avg.CATE']-cate_df['true.CATE'])
# sns.regplot(x='std.CATE',y='err.CATE',data=cate_df)
# sns.scatterplot(x='true.CATE',y='avg.CATE',size='std.CATE',data=cate_df)

m = pymalts.malts('Y','T',data=df_train, discrete=[], C=5,k=10)
res = m.fit()
print(res.x)

mg = m.get_matched_groups(df_est,50)


# cate_mean = m.CATE(mg,model='mean')
cate_linear = m.CATE(mg,model='linear')
# cate_RF = m.CATE(mg,model='RF')


fig, ax = plt.subplots()
sns.scatterplot(x='true.CATE',y='avg.CATE',size='std.CATE',hue='treatment',alpha=0.2,sizes=(10,200),data=cate_df)
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
err_malts_mean = [] #list( np.array(list( np.abs(t_true - cate_mean['CATE']) )) )
err_malts_linear = list(np.array(list( np.abs(t_true - cate_linear['CATE']) ))/ate_true )
err_malts_RF = [] #list(np.array(list( np.abs(t_true - cate_RF['CATE']) )))

label_malts = [ 'MALTS (mean)' for i in range(len(err_malts_mean)) ]+[ 'MALTS (linear)' for i in range(len(err_malts_linear)) ]+[ 'MALTS (RF)' for i in range(len(err_malts_RF)) ]
err_malts = err_malts_mean + err_malts_linear + err_malts_RF

label_malts = [ 'MALTS (Multifold)' for i in range(len(err_malts_mf)) ]
err_malts += err_malts_mf

'''
#----------------------------------------------------------------------------------------------
##Prognostic
prog = prognostic.prognostic('Y','T',df_train)
prog_mg = prog.get_matched_group(df_est) 

err_prog = list(np.array(list( np.abs(t_true - prog_mg['CATE']) ))/ate_true )
label_prog = [ 'Prognostic Score' for i in range(len(err_prog)) ]

#----------------------------------------------------------------------------------------------
##DBARTS

import rpy2
# from rpy2.robjects.vectors import StrVector
# import rpy2.robjects.packages as rpack
from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects
# import rpy2.robjects.lib as rlib
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
utils = importr('utils')

Xc = np.array(df_train.loc[df_train['T']==0,df_train.columns[0:num_covariates]])
Yc = np.array(df_train.loc[df_train['T']==0,'Y'])

Xt = np.array(df_train.loc[df_train['T']==1,df_train.columns[0:num_covariates]])
Yt = np.array(df_train.loc[df_train['T']==1,'Y'])
#
dbarts = importr('dbarts')
bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
y_c_hat_bart = np.array(bart_res_c.rx(8))
bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
y_t_hat_bart = np.array(bart_res_t.rx(8))
t_hat_bart = np.array(list(y_t_hat_bart - y_c_hat_bart)[0])
t_true_bart = t_true

err_bart = list( np.abs(t_hat_bart - t_true)/ate_true )
label_bart = [ 'BART' for i in range(len(err_bart)) ]

#----------------------------------------------------------------------------------------------
##Causal Forest

base = importr('base')

grf = importr('grf')
Ycrf = Y.reshape((len(Y),1))
Tcrf = T.reshape((len(T),1))
crf = grf.causal_forest(X,Ycrf,Tcrf)
tauhat = grf.predict_causal_forest(crf,Xtest)
tau_hat = tauhat['predictions']
t_hat_crf = np.array([tau_hat[i] for i in range(len(tauhat))])
t_true_crf = np.array(t_true)

err_crf = list( np.abs(t_hat_crf - t_true)/ate_true )
label_crf = [ 'Causal Forest' for i in range(len(err_crf)) ]



# #---------------------------------------------------------------------------------------------
##MATCHIT Setup
df_train_matchit = df_train
df_est_matchit = df_est
df_train['TE'] = df_true_train['TE']
df_est['TE'] = df_true_est['TE']
df_train_matchit.to_csv('non-constant-treatment-continuous.csv',index=False)
df_est_matchit.to_csv('test-non-constant-treatment-continuous.csv',index=False)

formula_cov = 'X0'
for j in range(1,num_covariates):
    formula_cov += '+X%d'%(j)

# #---------------------------------------------------------------------------------------------
##GenMatch
string = """
library('MatchIt')
mydata <- read.csv('test-non-constant-treatment-continuous.csv')
r <- matchit( T ~ %s, method = "genetic", data = mydata, replace = TRUE)
mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
hh <- mydata[as.numeric(names(r$match.matrix[,])),'Y']- mydata[as.numeric(r$match.matrix[,]),'Y']

mydata2 <- mydata
mydata2$T <- 1 - mydata2$T
r2 <- matchit( T ~ %s, method = "genetic", data = mydata2, replace = TRUE)
mtch2 <- mydata2[as.numeric(names(r2$match.matrix[,])),]
hh2 <- mydata2[as.numeric(r2$match.matrix[,]),'Y'] - mydata2[as.numeric(names(r2$match.matrix[,])),'Y']
"""%(formula_cov,formula_cov)

genmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
match = genmatch.mtch
match2 = genmatch.mtch2
t_true_genmatch = np.hstack((match['TE'],match2['TE']))
t_hat_genmatch = np.hstack((np.array(genmatch.hh),np.array(genmatch.hh2)))

err_genmatch = list( np.abs(t_hat_genmatch - t_true_genmatch)/ate_true )
label_genmatch = [ 'GenMatch' for i in range(len(err_genmatch)) ]


# #---------------------------------------------------------------------------------------------
##Propensity Score Nearest Neighbor
string = """
library('MatchIt')
mydata <- read.csv('test-non-constant-treatment-continuous.csv')
r <- matchit( T ~ %s, method = "nearest", data = mydata, replace = TRUE)
mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
hh <- mydata[as.numeric(names(r$match.matrix[,])),'Y']- mydata[as.numeric(r$match.matrix[,]),'Y']

mydata2 <- mydata
mydata2$T <- 1 - mydata2$T
r2 <- matchit( T ~ %s, method = "nearest", data = mydata2, replace = TRUE)
mtch2 <- mydata2[as.numeric(names(r2$match.matrix[,])),]
hh2 <- mydata2[as.numeric(r2$match.matrix[,]),'Y'] - mydata2[as.numeric(names(r2$match.matrix[,])),'Y']
"""%(formula_cov,formula_cov)

psnn = SignatureTranslatedAnonymousPackage(string, "powerpack")
match = psnn.mtch
match2 = psnn.mtch2
t_true_psnn =  np.hstack((match['TE'],match2['TE']))
t_hat_psnn = np.hstack((np.array(psnn.hh),np.array(psnn.hh2)))

err_psnn = list( np.abs(t_hat_psnn - t_true_psnn)/ate_true )
label_psnn = [ 'Propensity Score' for i in range(len(err_psnn)) ]

# #---------------------------------------------------------------------------------------------
##Full Matching
string = """
library('MatchIt')
mydata <- read.csv('test-non-constant-treatment-continuous.csv')
r <- matchit( T ~ %s, method = "full", data = mydata, replace = TRUE)
mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
hh <- mydata[as.numeric(names(r$match.matrix[,])),'Y']- mydata[as.numeric(r$match.matrix[,]),'Y']
"""%(formula_cov)

full = SignatureTranslatedAnonymousPackage(string, "powerpack")
match = full.mtch
t_true_full = match['TE']
t_hat_full = np.array(full.hh)

err_full = list( np.abs(t_hat_full - t_true_full)/ate_true )
label_full = [ 'Full Matching' for i in range(len(err_full)) ]


# #---------------------------------------------------------------------------------------------
'''
err = pd.DataFrame()
err['Relative CATE Error (percentage)'] = np.array(err_malts + err_bart + err_crf + err_genmatch + err_psnn + err_full + err_prog)*100
err['Method'] = label_malts + label_bart + label_crf + label_genmatch + label_psnn + label_full + label_prog

fig, ax = plt.subplots(figsize=(40,50))
sns.boxenplot(x='Method',y='Relative CATE Error (percentage)',data=err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/boxplot_multifold_malts.png')
 
fig, ax = plt.subplots(figsize=(40,50))
sns.violinplot(x='Method',y='Relative CATE Error (percentage)',data=err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig('Figures/violin_multifold_malts.png')

err.to_csv('Logs/CATE_Multifold_Est_Error_File.csv')
