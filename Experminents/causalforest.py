# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:25:47 2020

@author: Harsh
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri

rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

base = importr('base')

grf = importr('grf')

def causalforest(outcome,treatment,data,n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    gen_skf = skf.split(data,data[treatment])
    cate_est = pd.DataFrame()
    covariates = set(data.columns) - set([outcome,treatment])
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        Ycrf = df_train[outcome]
        Tcrf = df_train[treatment]
        X = df_train[covariates]
        Xtest = df_est[covariates]
        
        crf = grf.causal_forest(X,Ycrf,Tcrf)
        tauhat = grf.predict_causal_forest(crf,Xtest)
        t_hat_crf = np.array(tauhat[0])
        
        cate_est_i = pd.DataFrame(t_hat_crf, index=df_est.index, columns=['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est
        