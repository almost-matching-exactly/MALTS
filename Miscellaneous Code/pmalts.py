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



def PMALTS(X,Y,T,discrete_col=[]):
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
            cons_tup = ( list(temp_df.columns)[:-2], np.array(row)[:-2] )
            d_set_tuple.append(cons_tup)
    
    X_dummy = np.zeros((n,len(d_set_tuple)))
    for i in range(0,n):
        X_dummy[i,:] = np.array([ belongs(tup[0],tup[1],X_d[i,:]) for tup in d_set_tuple ]).astype('int')
    
    Xprime = np.hstack((X_c,X_dummy))
    n,m = Xprime.shape
    L = np.eye(m)
    dcov=np.arange(m_c,m)
    def distance(xi,xj,L,dcov):
        m_c = len(xi)-len(dcov)
        xic,xjc = xi[:m_c],xj[:m_c]
        xid,xjd = xi[m_c:],xj[m_c:]
        s = np.dot(xic-xjc,np.dot(L[:m_c,:m_c],xic-xjc))
        s += np.dot( (np.equal(xid,xjd).astype('int')), np.dot(L[m_c:,m_c:],(np.equal(xid,xjd).astype('int') ) ) )
        return s
    
    adknn = dml.distance_metric_learning(m,diag=True,dcov=dcov,K=10,L=L,distance=distance)
    adknn.optimize(Xprime,Y,T,soft=False,numbatch=max(n//500,1),method='COBYLA')
    Lstar = adknn.L
    return ((adknn,Xprime,Y,T,d_set_tuple,dcov),Lstar)
    
def PMALTS_Test(Xtest,model,discrete_col=[]):
    adknn,Xprime,Y,T,d_set_tuple,dcov = model
    n,m = Xtest.shape
    d_columns = discrete_col
    Xtest_d, Xtest_c = split_discrete(Xtest,d_columns)
    n_d,m_d = Xtest_d.shape
    n_c,m_c = Xtest_c.shape
    
    Xtest_dummy = np.zeros((n,len(d_set_tuple)))
    for i in range(0,n):
        Xtest_dummy[i,:] = np.array([ belongs(tup[0],tup[1],Xtest_d[i,:]) for tup in d_set_tuple ]).astype('int')
    
    Lstar = adknn.L
    fLC = lambda y: Lstar
        
    Xtestprime = np.hstack((Xtest_c,Xtest_dummy))
    dcate_cobyla, t_hat_cobyla = adknn.CATE(Xtestprime,Xprime,Y,T,fLC=fLC,fLT=fLC)
    ATE_cobyla = np.average(t_hat_cobyla)
    return dcate_cobyla,t_hat_cobyla,ATE_cobyla