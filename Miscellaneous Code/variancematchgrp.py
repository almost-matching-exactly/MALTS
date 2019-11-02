#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:32:47 2018

@author: harshparikh
"""

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
num_control = 1000
num_treated = 1000
num_cov_dense = 10
num_covs_unimportant = 0
numExample = num_control + num_treated
num_covariates = num_cov_dense+num_covs_unimportant
numVariable = num_covariates
var_array_C = []
var_array_T = []

for i in range(0,5):
    currenttime = time.asctime(time.localtime())
    currenttime = currenttime.replace(':','_')
    print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable))
    print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
    
        
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
    adknn = dml.distance_metric_learning(m,discrete=discrete)
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
    
    
    var_array_C += [np.average(np.array([ np.var(v[0][0],axis=0) for k,v in dcate_cobyla.items() ]),axis=0)]
    var_array_T += [np.average(np.array([ np.var(v[0][1],axis=0) for k,v in dcate_cobyla.items() ]),axis=0)]
    
    delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))

fig = figure(figsize=(10,10))
rcParams.update({'font.size': 22})
rcParams['lines.linewidth'] = 5
errorbar(x=np.arange(0,num_covariates),y=np.average(np.array(var_array_C),axis=0),yerr=np.std(np.array(var_array_C),axis=0))
errorbar(x=np.arange(0,num_covariates),y=np.average(np.array(var_array_T),axis=0),yerr=np.std(np.array(var_array_T),axis=0))
xlabel('Covariate Indices')
ylabel('Average Variance')
legend(['matched groups in control','matched groups in treated'],loc=4)
tight_layout()
fig.savefig('Figures/MALTS_Var_Match_Group_Plot_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
#        
log.close()