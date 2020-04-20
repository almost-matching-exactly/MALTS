#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:00:40 2020

@author: harshparikh
"""

import numpy as np
import pandas as pd

def construct_sec_order(arr):
    # an intermediate data generation function used for generating second order information
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
    return np.array(second_order_feature)


def data_generation_dense_endo(num_samples, num_cov_dense, num_covs_unimportant,rho=0):
    def u(x):
        T = []
        for row in x:
            l = ( 1 + np.tanh( ( row[0] + row[1] ) / 20 ) ) / 2
            t = np.random.binomial(1,l/2)
            T.append(t)
        return np.array(T)
    # the data generating function that we will use. include second order information
    mu = 1*np.ones((num_samples,num_cov_dense))
    sigma = (1-rho)*np.eye(num_cov_dense) + rho*np.ones((num_cov_dense,num_cov_dense))
    x = np.matmul(np.random.normal(0, 0.5, size=(num_samples, num_cov_dense)),sigma) + mu  #+ np.random.lognormal(1, 0.25, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
   
    errors = np.random.normal(0, 1, size=num_samples) #noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ dense_bs_sign[i]*10.*(1./2**(i+1)) for i in range(num_cov_dense) ]
    
    treatment_eff_coef = np.random.normal( 1.0, 0.5, size=num_cov_dense)
    treatment_effect = np.dot(x, treatment_eff_coef)
    second = construct_sec_order(x[:,:num_cov_dense])
    treatment_eff_sec = np.sum(second, axis=1)
    x2 = np.random.normal(1, 1.5, size=(num_samples, num_covs_unimportant))
    T = u(x)
    y0 = np.dot(x, np.array(dense_bs))  + errors 
    y1 = np.dot(x, np.array(dense_bs)) + (treatment_effect  + treatment_eff_sec) + errors 
    y = T*y1 + (1-T)*y0     # y for conum_treatedrol group 
    te = y1 - y0
     #+ np.random.normal(3, 1.5, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    
    df = pd.DataFrame(np.hstack([x, x2]), columns = list( ['X%d'%(j) for j in range(num_cov_dense + num_covs_unimportant)] ))
    df['Y'] = y
#    print((len(y),len(T))
    df['T'] = T
    
    df_true = pd.DataFrame()
    df_true['Y1'] = y1
    df_true['Y0'] = y0
    df_true['TE'] = te
    return df, df_true


def data_generation_dense_endo_2(num_samples, num_cov_dense, num_covs_unimportant,rho=0):
    def u(x):
        T = []
        for row in x:
            l = ( 1 + np.tanh( ( row[-1] + row[-2] - row[-3])  ) )
            t = np.random.binomial(1,l/2)
            T.append(t)
        return np.array(T)
    # the data generating function that we will use. include second order information
    mu = 1*np.ones((num_samples,num_cov_dense))
    sigma = (1-rho)*np.eye(num_cov_dense) + rho*np.ones((num_cov_dense,num_cov_dense))
    x = np.matmul(np.random.normal(0, 0.5, size=(num_samples, num_cov_dense)),sigma) + mu  #+ np.random.lognormal(1, 0.25, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
   
    errors = np.random.normal(0, 1, size=num_samples) #noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ dense_bs_sign[i]*10.*(1./2**(i+1)) for i in range(num_cov_dense) ]
    
    treatment_eff_coef = np.random.normal( 1.0, 0.5, size=num_cov_dense)
    treatment_effect = np.dot(x, treatment_eff_coef)
    second = construct_sec_order(x[:,:num_cov_dense])
    treatment_eff_sec = np.sum(second, axis=1)
    x2 = np.random.normal(1, 1.5, size=(num_samples, num_covs_unimportant))
    T = u(x2)
    y0 = np.dot(x, np.array(dense_bs))  + errors 
    y1 = np.dot(x, np.array(dense_bs)) + (treatment_effect  + treatment_eff_sec) + errors 
    y = T*y1 + (1-T)*y0     # y for conum_treatedrol group 
    te = y1 - y0
     #+ np.random.normal(3, 1.5, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    
    df = pd.DataFrame(np.hstack([x, x2]), columns = list( ['X%d'%(j) for j in range(num_cov_dense + num_covs_unimportant)] ))
    df['Y'] = y
#    print((len(y),len(T))
    df['T'] = T
    
    df_true = pd.DataFrame()
    df_true['Y1'] = y1
    df_true['Y0'] = y0
    df_true['TE'] = te
    return df, df_true

def data_generation_dense_mixed_endo(num_samples, num_cont_imp, num_disc_imp, num_cont_unimp,num_disc_unimp,rho=0):
    def u(x):
        T = []
        for row in x:
            l = ( 1 + np.tanh( ( row[0] + row[1] ) / 20 ) ) / 2
            t = np.random.binomial(1,l/2)
            T.append(t)
        return np.array(T)
    # the data generating function that we will use. include second order information
    xc = np.random.normal(1, 1.5, size=(num_samples, num_cont_imp)) #+ np.random.lognormal(1, 0.25, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
    xd = np.random.binomial(1, 0.5, size=(num_samples, num_disc_imp))   # data for conum_treatedrol group
    x = np.hstack((xc,xd))
    
    errors = np.random.normal(0, 1, size=num_samples)    # some noise
    
    num_cov_dense = num_cont_imp + num_disc_imp
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ dense_bs_sign[i]*np.random.normal(10,3) for i in range(num_cov_dense) ]
    
    treatment_eff_coef = np.random.normal( 1.0, 0.5, size=num_cov_dense)
    treatment_effect = np.dot(x, treatment_eff_coef)
    second = construct_sec_order(x[:,:num_cov_dense])
    treatment_eff_sec = np.sum(second, axis=1)
    xc2 = np.random.normal(1, 1.5, size=(num_samples, num_cont_unimp))
    xd2 = np.random.binomial(1, 0.5, size=(num_samples, num_disc_unimp))
    x2 = np.hstack((xc2,xd2))
    T = u(x)
    y0 = np.dot(x, np.array(dense_bs))  + errors 
    y1 = np.dot(x, np.array(dense_bs)) + (treatment_effect  + treatment_eff_sec) + errors 
    y = T*y1 + (1-T)*y0     # y for conum_treatedrol group 
    te = y1 - y0
     #+ np.random.normal(3, 1.5, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    num_covs_unimportant = num_cont_unimp + num_disc_unimp
    df = pd.DataFrame(np.hstack([x, x2]), columns = list( ['X%d'%(j) for j in range(num_cov_dense + num_covs_unimportant)] ))
    df['Y'] = y
#    print((len(y),len(T))
    df['T'] = T
    discrete = ['X%d'%(j) for j in range(num_cont_imp,num_cov_dense)] + ['X%d'%(j) for j in range(num_cov_dense + num_cont_unimp, num_cov_dense + num_covs_unimportant)]
    df_true = pd.DataFrame()
    df_true['Y1'] = y1
    df_true['Y0'] = y0
    df_true['TE'] = te
    return df, df_true, discrete

