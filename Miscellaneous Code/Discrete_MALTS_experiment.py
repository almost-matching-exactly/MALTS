# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:55:55 2018

@author: Harsh
"""
import numpy as np
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


K=15
currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('Logs/log'+currenttime+'.txt','w')

numExperiment = 1

num_control = 2000
num_treated = 2000
num_cov_dense = 10
num_covs_unimportant = 20
numExample = num_control + num_treated
num_covariates = num_cov_dense+num_covs_unimportant
numVariable = num_covariates

ATEcobylaArray = []
ATEnelmeadArray = []
ATEneuralnetArray = []

print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
#for i in range(0,numExperiment):
    #print "Experiment", i
    
#Data Generation
    

#non-constant treatment discrete
data = dg.data_generation_dense_discrete(num_control, num_treated, num_cov_dense, num_covs_unimportant)
df, dense_bs, treatment_eff_coef = data
X,Y,T = np.array(df[df.columns[0:num_covariates]],dtype=np.float64), np.array(df['outcome'],dtype=np.float64), np.array(df['treated'],dtype=np.float64)
n,m = X.shape
dftest,_,_ = dg.data_generation_dense_discrete(2500, 2500, num_cov_dense, num_covs_unimportant)
Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]],dtype=np.float64), np.array(dftest['outcome'],dtype=np.float64), np.array(dftest['treated'],dtype=np.float64)
t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
discrete = True


currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')

###############################
##TREATMENT EFFECT PREDICTION##
###############################

#InsaneLearning
insane = InsaneLearner.InsaneLearner(1000)
insane.insaneLearning(X,Y,T)
t_hat_insane = insane.CATE(Xtest)
#discrete = False
t_true_insane = t_true

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_insane), min(t_hat_insane)), max(max(t_true_insane), max(t_hat_insane)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_insane,t_hat_insane,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_insane_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')

#

##COBYLA Optimization
adknn_cobyla = dml.distance_metric_learning(m,discrete=discrete)
Dc,Dt = adknn_cobyla.split(X,Y,T)
Xc,Yc,Tc = Dc
Xt,Yt,Tt = Dt
optresult = adknn_cobyla.optimize(X,Y,T,numbatch=1)

Lstar_cobyla = adknn_cobyla.L
fLCcobyla = lambda y: Lstar_cobyla

dfC_cobyla = pd.DataFrame(Lstar_cobyla)
dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(Xtest,X,Y,T,fLC=fLCcobyla)

t_true_cobyla = t_true

print("L matrix", file=log)
print(dfC_cobyla, file=log)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_cobyla), min(t_hat_cobyla)), max(max(t_true_cobyla), max(t_hat_cobyla)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_cobyla,t_hat_cobyla,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_cobyla_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')


#Generic FLAME comparison 
#holdout,_ = dg.data_generation_gradual_decrease_discrete(num_control,num_treated,num_covariates,exponential=True)
resFlame = FLAME.run_bit(dftest, df, list(range(num_covariates)), [2]*num_covariates, tradeoff_param = 0)
resdf = resFlame[1]
t_true_flame = []
t_hat_flame = []
t_coeff_flame = list(treatment_eff_coef)+list(np.zeros((num_covs_unimportant,)))
for resdfi in resdf:
    if resdfi.size > 0:
        rescol = list(resdfi.columns[0:-2])
        rescol5 = [x for x in rescol if x<5]
        tcfi = [ t_coeff_flame[rescol[i]] for i in range(0,len(rescol)) ]
        for i in resdfi.index:
            t_hat_flame.append(resdfi.loc[i,'effect'])
            rescovi =  resdfi[resdfi.columns[0:-2]].iloc[i]
            soei = [[ resdfi[j].iloc[i] for j in rescol5 ]]
            t_true_flamei = np.dot(rescovi,tcfi) +  np.sum(dg.construct_sec_order(soei))
            t_true_flame.append(t_true_flamei)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_flame), min(t_hat_flame)), max(max(t_true_flame), max(t_hat_flame)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_flame,t_hat_flame,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_flame_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')


#Causal Random Forest
utils = importr('utils')
#packnames = ('MatchIt','grf')
#utils.install_packages(StrVector(packnames))
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

#
##GenMatch
df.to_csv('non-constant-treatment-continuous.csv',index=False)
dftest = pd.DataFrame(Xtest)
df.to_csv('test-non-constant-treatment-continuous.csv',index=False)
string = """
library('MatchIt')

mydata <- read.csv('test-non-constant-treatment-continuous.csv')

r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29,methods = 'genetic', data = mydata)

mtch <- mydata[as.numeric(names(r$match.matrix[,])),]

hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
"""
genmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
t_hat_genmatch = np.array(genmatch.hh)
x_hat_genmatch = np.array(genmatch.mtch).T[:,:30]
t_true_genmatch = np.dot((x_hat_genmatch[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_genmatch[:,:5]), axis=1)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_genmatch), min(t_hat_genmatch)), max(max(t_true_genmatch), max(t_hat_genmatch)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_genmatch,t_hat_genmatch,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_genmatch_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
#

string = """
library('MatchIt')

mydata <- read.csv('test-non-constant-treatment-continuous.csv')

r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29,methods = 'nearest', data = mydata)

mtch <- mydata[as.numeric(names(r$match.matrix[,])),]

hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
"""
nearestmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
t_hat_nearestmatch = np.array(nearestmatch.hh)
x_hat_nearestmatch = np.array(nearestmatch.mtch).T[:,:30]
t_true_nearestmatch = np.dot((x_hat_nearestmatch[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_nearestmatch[:,:5]), axis=1)

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_nearestmatch), min(t_hat_nearestmatch)), max(max(t_true_nearestmatch), max(t_hat_nearestmatch)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_nearestmatch,t_hat_nearestmatch,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_nearestmatch_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
#
#

#----------------------------------------------------------------------------------------------
##DBARTS
dbarts = importr('dbarts')
bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
y_c_hat_bart = np.array(bart_res_c.rx(8))
bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
y_t_hat_bart = np.array(bart_res_t.rx(8))
t_hat_bart = list(y_t_hat_bart - y_c_hat_bart)[0]
t_true_bart = t_true

fig = figure(figsize=(10,10))
identity_line = np.linspace(min(min(t_true_bart), min(t_hat_bart)), max(max(t_true_bart), max(t_hat_bart)))
plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
scatter(t_true_bart,t_hat_bart,alpha=0.15)
xlabel('True Treatment')
ylabel('Predicted Treatment')
fig.savefig('Figures/CATE_distancemetriclearning_BART_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
##
        
#
delt_insane = np.array(list(map(np.abs,np.array(t_hat_insane) - np.array(t_true_insane))))
delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))
delt_crf = np.array(list(map(np.abs,np.array(t_hat_crf) - np.array(t_true_crf))))
delt_flame = np.array(list(map(np.abs,np.array(t_hat_flame) - np.array(t_true_flame))))
delt_genmatch = np.array(list(map(np.abs,np.array(t_hat_genmatch) - np.array(t_true_genmatch))))
delt_nearestmatch = np.array(list(map(np.abs,np.array(t_hat_nearestmatch) - np.array(t_true_nearestmatch))))
#delt_bart = np.array(list(map(np.abs,np.array(t_hat_bart) - np.array(t_true_bart))))

#
fig = figure()
violinplot(delt_cobyla,positions=[1])
violinplot(delt_flame,positions=[2])
violinplot(delt_genmatch,positions=[3])
violinplot(delt_nearestmatch,positions=[4])
violinplot(delt_bart,positions=[5])
violinplot(delt_insane,positions=[6])
violinplot(delt_crf,positions=[7])
ylim(top=40) 
xticks([1,2,3,4,5,6,7],['MALTS','FLAME','Genmatch','NN Matching','Diff. of BARTS','Diff. of RF','Causal Forest'],rotation=45)
ylabel('CATE absolute error')
tight_layout()
fig.savefig('Figures/violinplot_error_cates_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')

log.close()
