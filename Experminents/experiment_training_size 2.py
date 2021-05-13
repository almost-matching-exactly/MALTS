#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:50:50 2019

@author: harshparikh
"""
import numpy as np
import pandas as pd
import pymalts
import dg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df_err = pd.DataFrame(columns=['training set size','num important cov','total num cov','training objective','test err'])
log = open('log_training_size.csv','w+')
print('TRAINING SIZE EXPERIMENT',file=log)
log.close()
df_err.to_csv('log_training_size.csv', mode='a', header=True)

for i in range(7):
    np.random.seed(0)
    num_cov_dense = 5
    num_covs_unimportant = 5
    n_est = 2000
    num_covariates = num_cov_dense+num_covs_unimportant
    df_est,_,_ = dg.data_generation_dense_endo(n_est, num_cov_dense, num_covs_unimportant,rho=0.2)
    df_est = df_est.drop(columns = 'matched')
    Xtest,Ytest,Ttest = np.array(df_est[df_est.columns[0:num_covariates]]), np.array(df_est['outcome']), np.array(df_est['treated'])
    
    for trial in range(10):
        numExample = 100 * (2**i)
        data = dg.data_generation_dense_endo(numExample, num_cov_dense, num_covs_unimportant,rho=0.2)
        df, dense_bs, treatment_eff_coef = data
        df_train = df.drop(columns = 'matched')
        X,Y,T = np.array(df_train[df_train.columns[0:num_covariates]]), np.array(df_train['outcome']), np.array(df_train['treated'])
        
        t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
        print((i,trial))
        
        #del Xtest,Ytest,Ttest,df,dense_bs, treatment_eff_coef
        
        m = pymalts.malts('outcome','treated',data=df_train, discrete=[], C=5,k=10)
        res = m.fit(method='COBYLA')
        
        mg = m.get_matched_groups(df_est,10)
        cate_linear = m.CATE(mg,model='linear')
        err_malts_linear = list( np.abs(t_true - cate_linear['CATE']) )
        
        df2 = pd.DataFrame({
            'training set size': [numExample],
            'num important cov': [num_cov_dense],
            'total num cov': [num_cov_dense+num_covs_unimportant],
            'training objective': [m.objective(m.M)],
            'relative test err': [np.mean(err_malts_linear)/np.mean(t_true)]
            })
        df2.to_csv('log_training_size.csv', mode='a', header=False)
        df_err = df_err.append(df2, ignore_index = True)
        print(df2)

fig = plt.figure(figsize=(15,15))
sns.lineplot(data=df_err,x='training set size',y='relative test err')
fig.savefig('training_set_size_1.png')