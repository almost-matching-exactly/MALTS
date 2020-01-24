#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:50:50 2019

@author: harshparikh
"""
import numpy as np
import pandas as pd
import malts
import dg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

numExample = 5000
num_cov_dense = 8
num_covs_unimportant = 10
n_est = 8000
num_covariates = num_cov_dense+num_covs_unimportant

data = dg.data_generation_dense_endo(numExample, num_cov_dense, num_covs_unimportant,rho=0.2)
df, dense_bs, treatment_eff_coef = data
df_train = df.drop(columns = 'matched')
X,Y,T = np.array(df_train[df_train.columns[0:num_covariates]]), np.array(df_train['outcome']), np.array(df_train['treated'])

df_est,_,_ = dg.data_generation_dense_endo(n_est, num_cov_dense, num_covs_unimportant,rho=0.2)
df_est = df_est.drop(columns = 'matched')

Xtest,Ytest,Ttest = np.array(df_est[df_est.columns[0:num_covariates]]), np.array(df_est['outcome']), np.array(df_est['treated'])
t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
print(dense_bs)

#del Xtest,Ytest,Ttest,df,dense_bs, treatment_eff_coef

m = malts.malts('outcome','treated',data=df_train, discrete=[], C=5,k=10)
res = m.fit(method='COBYLA')
print(res.x)

mg = m.get_matched_groups(df_est,50)

cate_mean = m.CATE(mg,model='mean')
cate_linear = m.CATE(mg,model='linear')
cate_RF = m.CATE(mg,model='RF')

fig, ax = plt.subplots()
plt.scatter(t_true,cate_linear['CATE'],alpha=0.01)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

err_malts_mean = list( np.array(list( np.abs(t_true - cate_mean['CATE']) )) )
err_malts_linear = list(np.array(list( np.abs(t_true - cate_linear['CATE']) ))[:,0])
err_malts_RF = list(np.array(list( np.abs(t_true - cate_RF['CATE']) )))

label_malts = [ 'MALTS (mean)' for i in range(len(err_malts_mean)) ]+[ 'MALTS (linear)' for i in range(len(err_malts_linear)) ]+[ 'MALTS (RF)' for i in range(len(err_malts_RF)) ]
err_malts = err_malts_mean + err_malts_linear + err_malts_RF

#----------------------------------------------------------------------------------------------
##DBARTS

import rpy2
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpack
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.lib as rlib
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
utils = importr('utils')

Xc = m.Xc_C
Yc = m.Y_C

Xt = m.Xc_T
Yt = m.Y_T
#
dbarts = importr('dbarts')
bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
y_c_hat_bart = np.array(bart_res_c.rx(8))
bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
y_t_hat_bart = np.array(bart_res_t.rx(8))
t_hat_bart = np.array(list(y_t_hat_bart - y_c_hat_bart)[0])
t_true_bart = t_true

err_bart = list( np.abs(t_hat_bart - t_true) )
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

err_crf = list( np.abs(t_hat_crf - t_true) )
label_crf = [ 'Causal Forest' for i in range(len(err_crf)) ]

err = pd.DataFrame()
err['CATE Abs. Error'] = err_malts + err_bart + err_crf
err['Method'] = label_malts + label_bart + label_crf

fig = plt.Figure(figsize=(15,15))
sns.catplot(x='Method',y='CATE Abs. Error',data=err,kind='boxen')
plt.xticks(rotation=65, horizontalalignment='right')
fig.savefig('boxplot_malts.png')

fig = plt.Figure(figsize=(15,15))
sns.catplot(x='Method',y='CATE Abs. Error',data=err,kind='violin')
plt.xticks(rotation=65, horizontalalignment='right')
fig.savefig('violin_malts.png')

# #---------------------------------------------------------------------------------------------
# ##GenMatch
# df.to_csv('non-constant-treatment-continuous.csv',index=False)
# #dftest = pd.DataFrame(Xtest)
# df_est.to_csv('test-non-constant-treatment-continuous.csv',index=False)

# string = """
# library('MatchIt')

# mydata <- read.csv('test-non-constant-treatment-continuous.csv')

# r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17,
#              method = "genetic", data = mydata)

# mtch <- mydata[as.numeric(names(r$match.matrix[,])),]

# hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
# """
# genmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
# t_hat_genmatch = np.array(genmatch.hh)
