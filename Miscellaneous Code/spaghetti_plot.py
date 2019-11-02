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
num_n = 5
num_p = 5
n_array = [200,700,1400,2700,5000]
p_imp_array = np.linspace(4,20,num_p)
delt_mat = np.zeros((num_n,num_p))
reldelt_mat = np.zeros((num_n,num_p)) 
ate_mat = np.zeros((num_n,num_p)) 
#p_nonimp_array = np.linspace(2,50,10)

for i in range(0,num_p):
    for j in range(0,num_n):
        currenttime = time.asctime(time.localtime())
        currenttime = currenttime.replace(':','_')
        num_control = int(n_array[j])//2
        num_treated = int(n_array[j])//2
        num_cov_dense = int(p_imp_array[i])
        num_covs_unimportant = 0
        numExample = num_control + num_treated
        num_covariates = num_cov_dense+num_covs_unimportant
        numVariable = num_covariates
        print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable))
        print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
        numExperiment += 1
            
        ##non-constant treatment continuous
        data = dg.data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant)
        df, dense_bs, treatment_eff_coef = data
        X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
        n,m = X.shape
        dftest,_,_ = dg.data_generation_dense_2(2500, 2500, num_cov_dense, num_covs_unimportant)
        Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])
        t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
        discrete = False

        
        currenttime = time.asctime(time.localtime())
        currenttime = currenttime.replace(':','_')
     
        
        #------------------------------------------------------------------------------------
        #MALTS Non-Gradient Optimization
        discrete = True
        adknn = dml.distance_metric_learning(m,diag=discrete)
        Dc,Dt = adknn.split(X,Y,T)
        Xc,Yc,Tc = Dc
        Xt,Yt,Tt = Dt
#        minUnitInBatch = 1000
#        numbatch = max(1,numExample // minUnitInBatch)
        optresult = adknn.optimize(X,Y,T,numbatch=1)
        
        Lstar = adknn.L
        fLC = lambda y: Lstar
        
        dfC = pd.DataFrame(Lstar)
        dcate_cobyla, t_hat_cobyla = adknn.CATE(Xtest,X,Y,T,fLC=fLC,fLT=fLC)
        t_true_cobyla = t_true
        print("L matrix", file=log)
        print(dfC, file=log)
        
        
#        fig = figure(figsize=(10,10))
#        identity_line = np.linspace(min(min(t_true_cobyla), min(t_hat_cobyla)), max(max(t_true_cobyla), max(t_hat_cobyla)))
#        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
#        scatter(t_true_cobyla,t_hat_cobyla,alpha=0.15)
#        xlabel('True Treatment')
#        ylabel('Predicted Treatment')
#        fig.savefig('Figures/CATE_distancemetriclearning_MALTS_NonGradient_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
        
        delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))
        delt_mat[j,i] = np.average(delt_cobyla)
        ate_mat[j,i] = np.average(t_hat_cobyla)
        reldelt_mat[j,i] = delt_mat[j,i]/ate_mat[j,i]

fig = figure(figsize=(8.75,7))
rcParams.update({'font.size': 22})
rcParams['lines.linewidth'] = 4
for row in delt_mat:
    plot(p_imp_array,row)
 
xlabel('Number of Covariates')
ylabel('CATE Absolute Error')
legend(['n='+str(n_array[i]) for i in range(0,num_n)])
tight_layout() 
fig.savefig('Figures/MALTS_Spaghetti_Plot.jpg')
        
log.close()