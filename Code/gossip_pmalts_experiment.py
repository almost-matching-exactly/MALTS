#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:14:15 2018

@author: harshparikh
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
import pmalts
import cfr_net_train
#import ACIC_2
import warnings

warnings.filterwarnings('ignore')

K=10
currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('Logs/log_mixed_'+currenttime+'.txt','w')

numExperiment = 1
n_array = [1000]
num_control = int(n_array[0])//2
num_treated = int(n_array[0])//2
num_cont_imp, num_disc_imp, num_cont_unimportant, num_disc_unimportant = 2, 2, 2, 2
num_covariates = num_cont_imp + num_disc_imp + num_cont_unimportant + num_disc_unimportant
numVariable = num_covariates
numExample = num_control + num_treated

print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable)+'_'+str(currenttime), file=log)
    
#Data Generation
    

##non-constant treatment mixed
data = dg.data_generation_dense_mixed(num_control, num_treated, num_cont_imp, num_disc_imp, num_cont_unimportant, num_disc_unimportant)
df, dense_bs, dense_bs_d, treatment_eff_coef, treatment_eff_coef_d  = data

X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
n,m = X.shape

dftest,_,_,_,_ = dg.data_generation_dense_mixed(2500, 2500, num_cont_imp, num_disc_imp, num_cont_unimportant, num_disc_unimportant)
Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])

t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cont_imp]), axis=1) + np.dot((Xtest[:,len(treatment_eff_coef):(len(treatment_eff_coef)+len(treatment_eff_coef_d))]),treatment_eff_coef_d)
discrete = True
gossips = np.linspace(0,1,10)

err_array = []
for gossip in gossips:
    #------------------------------------------------------------------------------------
    #PMALTS
    dcate_pmalts, t_hat_pmalts, ate_pmalts = pmalts.PMALTS(X,Y,T,Xtest,discrete_col=list(np.arange(num_cont_imp,num_cont_imp+num_disc_imp+num_disc_unimportant)),gossip=gossip)
    t_true_pmalts = t_true
    delt_pmalts = np.array(list(map(np.abs,np.array(t_hat_pmalts) - np.array(t_true_pmalts))))
    err_array.append(np.average(delt_pmalts))
    
fig = figure(figsize=(10,10))
rcParams.update({'font.size':36})
plot(gossips, err_array, color="black", linestyle="dashed", linewidth=3.0)
ylabel('Absolute CATE Error')
xlabel('Gossip Ratio')
tight_layout()
fig.savefig('Figures/Gossip_PMALTS_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cont_imp)+'_'+str(num_disc_imp)+'_'+str(num_cont_unimportant)+'_'+str(num_disc_unimportant)+'_'+str(currenttime)+'.jpg')
log.close()