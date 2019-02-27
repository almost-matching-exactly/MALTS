#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:28:20 2018

@author: harshparikh

"""

import numpy as np
from matplotlib.pyplot import *
import pandas as pd
import distancemetriclearning as dml
import FLAMEbit as FLAME

import warnings
warnings.filterwarnings('ignore')

def split_discrete(X,d_columns):
    if len(d_columns)>0:
        n,m = X.shape
        c_columns = set(np.arange(0,m)) - set(d_columns)
        X_discrete = np.array([X[:,i] for i in d_columns])
        X_continuous = np.array([X[:,i] for i in c_columns])
        X_discrete = X_discrete.T
        X_continuous = X_continuous.T
        return X_discrete, X_continuous
    else:
        return None, X

def club_discrete(X,d_columns):
    n,m = X.shape
    X_discrete, X_continuous = split_discrete(X,d_columns)
    d = {}
    for i in range(0,n):
        if str(X_discrete[i,:]) not in d:
            d[str(X_discrete[i,:])] = [],[],[]
        d[str(X_discrete[i,:])] = d[str(X_discrete[i,:])] + [X_continuous[i,:]]
    return d

def belongs(colid,s1,s2):
    s21 = [ s2[i] for i in colid ]
    return np.array_equal(s1,s21)

def diameterMatchGroup(tup,L):
    xi = tup[2]
    Xs = np.vstack((tup[0][0],tup[0][1]))
    l_dis = [ np.dot(xi-xj,np.dot(L,xi-xj)) for xj in Xs]
    return max(l_dis)



def PMALTS(X,Y,T,discrete_col=[],gossip=0):
    Xb,Yb,Tb = X,Y,T
    n,m = Xb.shape
    d_columns = discrete_col
    X_d, X_c = split_discrete(Xb,d_columns)
    n_d,m_d = X_d.shape
    n_c,m_c = X_c.shape
    df_d = pd.DataFrame(X_d)
    df_d['outcome'] = Yb
    df_d['treated'] = Tb
    df_d['matched'] = np.zeros((n_d,))
    
    resFlame = FLAME.run_bit(df_d, df_d, list(range(m_d)), [2]*m_d, tradeoff_param = 1)
    resdf = resFlame[1]
    d_set_tuple = []
    for temp_df in resdf:
        for index, row in temp_df.iterrows():
            cons_tup = ( list(temp_df.columns)[:-2], np.array(row)[:-2], [], [], [], dml.distance_metric_learning(m-m_d,discrete=True,K=10))
            d_set_tuple.append(cons_tup)
    
    for i in range(0,n):
        for j in range(0,len(d_set_tuple)):
            tup = d_set_tuple[j]
            if belongs(tup[0],tup[1],X_d[i,:]):
                d_set_tuple[j] = (tup[0],tup[1],tup[2]+[X_c[i,:]],tup[3]+[Yb[i]],tup[4]+[Tb[i]],tup[5])
    
    d_set_tuple_pruned = []
    default_tup = ([],np.array([]),X_c, Yb, Tb, dml.distance_metric_learning(m-m_d,discrete=True,K=10))
    for tup in d_set_tuple:
        if len(tup[3])>(tup[5].K)*2:
            d_set_tuple_pruned += [tup]
    d_set_tuple_pruned += [default_tup]
    
    d_set_tuple = d_set_tuple_pruned
    
    for j in range(0,len(d_set_tuple)):
        print("Tuple-"+str(j))
        d_set_tuple[j][5].optimize(np.array(d_set_tuple[j][2]),np.array(d_set_tuple[j][3]),np.array(d_set_tuple[j][4]),numbatch = max(1,len(d_set_tuple[j][3])//500 ))
        
    #gossip update
    Larray = []
    szarray = []
    for j in range(0,len(d_set_tuple)):
        tup = d_set_tuple[j]
        Larray.append(np.copy(tup[5].L))
        szarray.append(len(tup[3]))
    
    Larray = np.array(Larray)
    szarray = np.array(szarray)
    for j in range(0,len(d_set_tuple)):
        tup = d_set_tuple[j]
        gossip = np.sum(np.array(list(map(lambda x,y: x*y, szarray,Larray))),axis=0)
        alpha = gossip
        beta = (1-gossip)
        d_set_tuple[j][5].L = ( alpha*gossip + beta*tup[5].L)/(alpha+beta)
    return d_set_tuple

def PMALTS_Test(Xtest,d_set_tuple,d_columns=[]):
    n,m = Xtest.shape
    ntest,mtest = Xtest.shape
    X_test_d, X_test_c = split_discrete(Xtest,d_columns)
    t_hat_cobyla = []
    t_hat_index = []
    w_hat_cobyla = []
    dcate_cobyla = {}
    for i in range(0,n):
        t_hat_i = []
        w_hat_i = []
        dcate_i = {}
        for j in range(0,len(d_set_tuple)):
            tup = d_set_tuple[j]
            if belongs(tup[0],tup[1],X_test_d[i,:]):
                Lstar = tup[5].L
                fL = lambda y: Lstar
                dcate_ij, t_hat_ij = tup[5].CATE(np.array([X_test_c[i,:]]),np.array(tup[2]),np.array(tup[3]),np.array(tup[4]),fL)
                w_ij = np.exp(10*len(tup[1])**2)*(1/(diameterMatchGroup(dcate_ij[0],Lstar)**0))
                #w_ij = 1/(diameterMatchGroup(dcate_ij[0],Lstar)**0)
                t_hat_i += [t_hat_ij[0]*w_ij]
                w_hat_i += [w_ij]
                dcate_i[j] = dcate_ij[0]
        val = np.average(t_hat_i)/np.average(w_hat_i)
        if not np.isnan(val):
            t_hat_cobyla.append( np.average(t_hat_i)/np.average(w_hat_i) )
            w_hat_cobyla.append(np.min(w_hat_i))
            dcate_cobyla[i] = dcate_i
            t_hat_index.append(i)
    ATE_cobyla = np.average(t_hat_cobyla)
    return dcate_cobyla,t_hat_cobyla,ATE_cobyla,t_hat_index