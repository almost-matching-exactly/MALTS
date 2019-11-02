#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:49:29 2018

@author: harshparikh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:55:55 2018

@author: Harsh
"""
import numpy as np
import scipy.optimize as opt
from sklearn import cluster as cluster
from matplotlib.pyplot import *
import time
import pandas as pd
import data_generation as dg
import distancemetriclearning as dml
import FLAMEbit as FLAME
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
import InsaneLearner
#import ACIC_2
import warnings
warnings.filterwarnings('ignore')

K=15
currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('Logs/log'+currenttime+'.txt','w')

numExperiment = 1

num_control = 500
num_treated = 500
num_cov_dense = 5
num_covs_unimportant = 5
numExample = num_control + num_treated
num_covariates = num_cov_dense+num_covs_unimportant
numVariable = num_covariates

print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
#for i in range(0,numExperiment):
    #print "Experiment", i
    
#Data Generation
    
#ACIC
#X,Y,T  = ACIC_2.normX, ACIC_2.Y, ACIC_2.T
#n,m = X.shape
#Xtest = X
#Xb,Yb,Tb = X[:1000,:], Y[:1000],T[:1000]
#discrete = True
#t_true = Y1 - Y0
#n,m = X.shape



#LaLonde Experiment
#df = pd.DataFrame.from_csv('lalonde.csv')
#X,Y,T = np.array(df[df.columns[1:8]]),np.array(df['re78']),np.array(df['treat'])
#n,m = X.shape
#discrete = True
#adknn_cobyla = dml.distance_metric_learning(7,discrete=discrete)
##optresult = adknn_cobyla.optimize(X,Y,T,numBatch=1)
#optresult = adknn_cobyla.optimize_parallel(X,Y,T,iterations=5,numbatch=10) #int(np.sqrt(numExample))
#LstarC_cobyla = adknn_cobyla.Lc
#LstarT_cobyla = adknn_cobyla.Lt
#fLCcobyla = lambda y: LstarC_cobyla
#fLTcobyla = lambda y: LstarT_cobyla
#dfC_cobyla = pd.DataFrame(LstarC_cobyla)
#dfT_cobyla = pd.DataFrame(LstarT_cobyla)
#dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(X,X,Y,T,fLC=fLCcobyla,fLT=fLTcobyla)
#dcobyla = adknn_cobyla.nearestneighbormatching(X,Y,T,fLCcobyla,fLTcobyla)
#ATEcobyla = adknn_cobyla.ATE(dcobyla)

#constant treatment continuous
#data = dg.data_generation_gradual_decrease(num_control,num_treated,num_covariates,exponential=True)
#df, dense_bs = data
#X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
#n,m = X.shape
#dftest,_ = dg.data_generation_gradual_decrease(2500,2500,num_covariates,exponential=True)
#Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])
#t_true = 10*np.ones( (len(Xtest),) )
#discrete = False
    
#constant treatment discrete
#data = dg.data_generation_gradual_decrease_discrete(num_control,num_treated,num_covariates,exponential=True)
#df, dense_bs = data
#X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
#n,m = X.shape
#Xtest = np.random.binomial(1, 0.5, size=(5000, num_covariates))
#t_true = 10*np.ones( (len(Xtest),) )
#discrete = True

##non-constant treatment continuous
data = dg.data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant)
df, dense_bs, treatment_eff_coef = data
X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
n,m = X.shape
dftest,_,_ = dg.data_generation_dense_2(2500, 2500, num_cov_dense, num_covs_unimportant)
Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])
t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
discrete = False

##non-constant treatment discrete
#data = dg.data_generation_dense_discrete(num_control, num_treated, num_cov_dense, num_covs_unimportant)
#df, dense_bs, treatment_eff_coef = data
#X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
#n,m = X.shape
#dftest,_,_ = dg.data_generation_dense_discrete(2500, 2500, num_cov_dense, num_covs_unimportant)
#Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])
#t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:5]), axis=1)
#discrete = True

currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')

#---------------------------------------------
#MALTS Non-Gradient Optimization
discrete = True
adknn = dml.distance_metric_learning(m,discrete=discrete)
Dc,Dt = adknn.split(X,Y,T)
Xc,Yc,Tc = Dc
Xt,Yt,Tt = Dt
optresult = adknn.optimize(X,Y,T,numbatch=4)

Lstar = adknn.L
fLC = lambda y: Lstar

dfC = pd.DataFrame(Lstar)
dcate_cobyla, t_hat_cobyla = adknn.CATE(Xtest,X,Y,T,fLC=fLC,fLT=fLC)
t_true_cobyla = t_true
print("L matrix", file=log)
print(dfC, file=log)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_cobyla), min(t_hat_cobyla)), max(max(t_true_cobyla), max(t_hat_cobyla)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_cobyla,t_hat_cobyla,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/LambdaExp_MALTS_NonGradient_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')

delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))

#---------------------------------------------
#MALTS Gradient Optimization
array = [0,5,10,20,50,100,200,500,700,800,1000,1200,1500,1700,1800,2000,2500,3000,5000]
Dc,Dt = adknnGD.split(X,Y,T)
Xc,Yc,Tc = Dc
Xt,Yt,Tt = Dt
errors2 = []
for i in array:
    discrete = True
    adknnGD = dml.distance_metric_learning(m,discrete=discrete)
    adknnGD.L = Lstar
    optresult = adknnGD.optimizeGD(X,Y,T,regfunc = lambda x: np.linalg.norm(x,ord='fro'),regcost=i,numbatch=4)
    LstarGD = adknnGD.L
    fLCGD = lambda y: LstarGD
    dfCGD = pd.DataFrame(LstarGD)
    dcate_GD, t_hat_GD = adknnGD.CATE(Xtest,X,Y,T,fLC=fLCGD,fLT=fLCGD)
    t_true_GD = t_true
    print("L matrix", file=log)
    print(dfCGD, file=log)
    delt_GD = np.array(list(map(np.abs,np.array(t_hat_GD) - np.array(t_true_GD))))
    errors2.append(np.average(delt_GD))

errors1 = []
for i in array:
    discrete = True
    adknnGD = dml.distance_metric_learning(m,discrete=discrete)
    adknnGD.L = Lstar
    optresult = adknnGD.optimizeGD(X,Y,T,regfunc = lambda x: np.linalg.norm(x,ord=1),regcost=i,numbatch=4)
    LstarGD = adknnGD.L
    fLCGD = lambda y: LstarGD
    dfCGD = pd.DataFrame(LstarGD)
    dcate_GD, t_hat_GD = adknnGD.CATE(Xtest,X,Y,T,fLC=fLCGD,fLT=fLCGD)
    t_true_GD = t_true
    print("L matrix", file=log)
    print(dfCGD, file=log)
    delt_GD = np.array(list(map(np.abs,np.array(t_hat_GD) - np.array(t_true_GD))))
    errors1.append(np.average(delt_GD))
    
fig = figure(figsize=(10,10))
plot(array,errors2)
plot(array,errors1)
hlines(np.average(delt_cobyla),0,3000)
xlabel('Lambda _ Reg Cost')
ylabel('Absolute Error')
fig.savefig('Figures/LambdaExp_MALTS_Gradient_abs_error'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')