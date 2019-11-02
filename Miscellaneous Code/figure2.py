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
import neuralnet
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
num_cov_dense = 2
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

##InsaneLearning
insane = InsaneLearner.InsaneLearner(1000)
insane.insaneLearning(X,Y,T)
t_hat_insane = insane.CATE(Xtest)
discrete = False
t_true_insane = t_true

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_insane), min(t_hat_insane)), max(max(t_true_insane), max(t_hat_insane)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_insane,t_hat_insane,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_insane_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')



#COBYLA Optimization
discrete = True
adknn_cobyla = dml.distance_metric_learning(m,discrete=discrete)
optresult = adknn_cobyla.optimize(X,Y,T,numBatch=1)
#optresult = adknn_cobyla.optimize_parallel(X,Y,T,iterations=5,numbatch=10) #int(np.sqrt(numExample))
LstarC_cobyla = adknn_cobyla.Lc
LstarT_cobyla = adknn_cobyla.Lt
fLCcobyla = lambda y: LstarC_cobyla
fLTcobyla = lambda y: LstarT_cobyla
#dcobyla = adknn_cobyla.nearestneighbormatching(X,Y,T,fLcobyla)
#ATEcobyla = adknn_cobyla.ATE(dcobyla)
dfC_cobyla = pd.DataFrame(LstarC_cobyla)
dfT_cobyla = pd.DataFrame(LstarT_cobyla)
dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(Xtest,X,Y,T,fLC=fLCcobyla,fLT=fLTcobyla)
#dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(X,X,Y,T,fLcobyla)
t_true_cobyla = t_true
print("L matrix Control", file=log)
print(dfC_cobyla, file=log)
print("L matrix Treated", file=log)
print(dfT_cobyla, file=log)
Lstar = (LstarC_cobyla + LstarT_cobyla)/2
print("L matrix averaged", file=log)
print(pd.DataFrame(Lstar), file=log)


fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_cobyla), min(t_hat_cobyla)), max(max(t_true_cobyla), max(t_hat_cobyla)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_cobyla,t_hat_cobyla,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_cobyla_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')


#Causal Random Forest
utils = importr('utils')
##packnames = ('MatchIt','grf')
##utils.install_packages(StrVector(packnames))
base = importr('base')
grf = importr('grf')
Ycrf = Y.reshape((len(Y),1))
Tcrf = T.reshape((len(T),1))
crf = grf.causal_forest(X,Ycrf,Tcrf)
tauhat = grf.predict_causal_forest(crf,Xtest)
tau_hat = tauhat[0]
t_hat_crf = list(tau_hat)
t_true_crf = t_true
#
fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_crf), min(t_hat_crf)), max(max(t_true_crf), max(t_hat_crf)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_crf,t_hat_crf,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_crf_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')


#GenMatch
df.to_csv('non-constant-treatment-continuous.csv',index=False)
#dftest = pd.DataFrame(Xtest)
dftest.to_csv('test-non-constant-treatment-continuous.csv',index=False)
string = """
library('MatchIt')

mydata <- read.csv('test-non-constant-treatment-continuous.csv')

r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6,
             methods = 'genetic', data = mydata)

mtch <- mydata[as.numeric(names(r$match.matrix[,])),]

hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
"""
genmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
t_hat_genmatch = np.array(genmatch.hh)
x_hat_genmatch = np.array(genmatch.mtch).T[:,:7]
t_true_genmatch = np.dot((x_hat_genmatch[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_genmatch[:,:5]), axis=1)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_genmatch), min(t_hat_genmatch)), max(max(t_true_genmatch), max(t_hat_genmatch)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_genmatch,t_hat_genmatch,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_genmatch_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')



delt_insane = np.array(list(map(np.abs,np.array(t_hat_insane) - np.array(t_true_insane))))
delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))
delt_crf = np.array(list(map(np.abs,np.array(t_hat_crf) - np.array(t_true_crf))))
delt_genmatch = np.array(list(map(np.abs,np.array(t_hat_genmatch) - np.array(t_true_genmatch))))


fig = figure()
violinplot(delt_cobyla,positions=[1])
violinplot(delt_crf,positions=[2])
violinplot(delt_insane,positions=[3])

xticks([1,2,3],['MALTS','Causal Forest','Insane']) #,'FLAME','Nelder-Mead','SQP'])
ylabel('Absolute Error in prediction of treatment effect')
fig.savefig('Figures/violinplot_error_cates_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')

log.close()

fig = figure()
violinplot(t_hat_cobyla,positions=[1])
violinplot(t_hat_insane,positions=[4])
violinplot(t_hat_crf,positions=[2])
violinplot(t_hat_genmatch,positions=[3])
xticks([1,2,3,4],['MALTS','Causal Forest','GenMatch','Insane'])
fig.savefig('Figures/violinplot_cates_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')

def prettyprintmatches(d):
    for k,v in d.iteritems():
        print(k)