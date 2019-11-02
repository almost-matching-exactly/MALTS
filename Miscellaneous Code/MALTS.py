# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:13:53 2018

@author: Harsh
"""

import numpy as np
import scipy.optimize as opt
import sklearn.linear_model as sklm
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
import multiprocessing


X = np.array(None)
Y = np.array(None)
T = np.array(None)
Dparts = []
dist_mat = {}

Xtest = np.array(None)
Ytest = np.array(None)
Ttest = np.array(None)
DpartsTest = []
dist_mat_test = {}

def objl(args):
    i,c,L,K = args
    Xc,Yc,Tc = Dparts[c]
#    di = 1/(dist_mat[c][i]+np.finfo(float).eps)
#    delyi = np.abs(Yc-Yc[i])
    kni, ykni = kneighbors(i,c,test=False,K=K) #slowstep
#    w = np.ones( (K,) ) #np.array(list(map( lambda x: np.exp(-1*(0)*self.distance(X[i,:],x,L)), kni ))) 
    return np.square( Yc[i] - (1.0/K)*np.sum(ykni) )

def kneighbors(i,c,test,K):
    if test:
        arr = dist_mat_test[c][i,:]
        indices = arr.argsort()[:K]
        Xc, Yc, Tc = DpartsTest[c]
        kn, ykn = Xc[indices,:], Yc[indices]
    if not test:
        arr = dist_mat[c][i,:]
        indices = arr.argsort()[:K]
        Xc, Yc, Tc = Dparts[c]
        kn, ykn = Xc[indices,:], Yc[indices]
    return kn, ykn

def makeMatchedGroup(args):
    i,L,k = args
    mg_i = ([],[],[])
    parts = len(DpartsTest)
    for c in range(0,parts):
        Xc,Yc,Tc = DpartsTest[c]
        if len(Tc)>0:
            neighborsX, neighborsY = kneighbors(i,c,test=True,K=k)
            neighborsT = [Tc[0]]*k
        if len(neighborsY) > 0:
            mg_i = ( mg_i[0] + list(neighborsX), mg_i[1] + list(neighborsY), mg_i[2] + list(neighborsT))
    mg_i = (np.array(mg_i[0]),np.array(mg_i[1]),np.array(mg_i[2]))
    return mg_i

def ITE(args):
    (i,di,tp,is_y_continuous) = args
    (x_i,y_i,t_i) = di
    if len(np.unique(y_i))<2:
        return np.zeros( (tp,1) )
    else:
        if is_y_continuous:
            ols = sklm.LinearRegression()
            olsres = ols.fit(t_i,y_i)
            beta = np.array(olsres.coef_)
            return beta
        else:
            lr = sklm.LogisticRegression()
            lrres = lr.fit(t_i,y_i)
            beta = lrres.coef_
            return beta

def calc_pairwise_d(X1,X2,L,test=True):
    n,m = X1.shape
    n1,_ = X2.shape
    dist = np.zeros((n,n1))
    for i in range(0,n):
        for j in range(0,n1):
            dist[i,j] = distances.d1(X1[i,:],X2[j,:],L)
    return dist
            
class distances:
    def d2(xi,xj,L,dcov=[]):
        p = len(xi)
        s = np.dot(xi-xj,np.dot(L[:p,:p],xi-xj))
        xi2d, xj2d, Ld = list(xi[dcov]),list(xj[dcov]),L[p:,p:]
        for i in range(0,len(dcov)):
            for j in range(0,i):
                if i!=j:
                    xi2d.append( str(xi[dcov[i]])+str(xi[dcov[j]]) )
                    xj2d.append( str(xj[dcov[i]])+str(xj[dcov[j]]) )
        c = np.array(np.array(xi2d)==np.array(xj2d)).astype('int')
        s += np.dot( c , np.dot(Ld,c) )
        return s
    
    def d(xi,xj,L,dcov=None):
        if dcov is None:
            dcov = []
        p = len(xi) - len(dcov)
        ccov = list(set(np.arange(0,len(xi))) - set (dcov))
        xic,xjc,Lc = xi[ccov],xj[ccov], (L[:p,:p])*(L[:p,:p])
        s = np.dot(xic-xjc,np.dot(Lc,xic-xjc))
        xid,xjd,Ld = xi[dcov],xj[dcov],(L[p:,p:])*(L[p:,p:])
        s += np.dot( (1 - np.equal(xid,xjd).astype('int')), np.dot(Ld,(1 - np.equal(xid,xjd).astype('int') ) ) )
        return s
    
    def d1(xi,xj,L):
        return np.dot(xi-xj,np.dot(L**2,xi-xj))
        
    
class distance_metric_learning:
    def __init__(self,number_of_covariates,diag=True,dcov=[],K=10,L=None,distance=distances.d):
        self.discrete = diag
        if L is None:
            self.L = np.eye(number_of_covariates)
        else:
            self.L = L
        self.K = K
        self.num_covariates = number_of_covariates
        self.dcov = dcov
        self.distance = lambda xi,xj,Lij: distance(xi,xj,Lij,self.dcov)

    
    def objective(self,c,L):
        #num_cores = multiprocessing.cpu_count()
        global dist_mat
        Xc, Yc, Tc = Dparts[c]
        nc = len(Xc)
        dist_mat[c] = calc_pairwise_d(Xc,Xc,L,test=False)
        K = self.K
        if nc<K:
            return 0
        nargs = [(i,c,L,K) for i in range(0,nc)]
        obj = 0
        objarray = np.array(Parallel(n_jobs=(1+(nc//10)),prefer="threads")(delayed(objl)(args) for args in nargs))

        obj = np.sum(objarray)
        diff = np.linalg.norm(L,ord='nuc')
        c_coeff = 0.00001
        cost = c_coeff*diff
        return obj + cost
    
    def translate(self,a):
        s = 0
        for i in a:
            s = 2*s + i
        return int(s)
    
    def split(self,X1,Y1,T1):
        if len(T1.shape)==1:
            tp = 1
        else:
            tp = T1.shape[1]
        parts = 2**tp
        n,m = X1.shape
        T1 = np.reshape(T1,(n,tp))
        dparts = [ ([],[],[]) for itr in range(0,parts) ]
        for i in range(0,n):
            part_indx = self.translate(T1[i,:])
            Xc, Yc, Tc = dparts[part_indx]
            Xc.append(X1[i,:])
            Yc.append(Y1[i])
            Tc.append(T1[i])
            dparts[part_indx] = (Xc,Yc,Tc)
        for i in range(0,parts):
            part = dparts[i]
            dparts[i] = (np.array(part[0]),np.array(part[1]),np.array(part[2]))
        return dparts
                  
    def optimize(self,X1,Y1,T1,soft=False,numbatch=1,method='COBYLA'):
        global X, Y, T, Dparts
        X = X1
        Y = Y1
        T = T1
        n,m = X.shape
        self.L = np.linalg.inv(np.cov(X,rowvar=False))
        if len(T.shape)==1:
            tp = 1
        else:
            tp = T.shape[1]
        parts = 2**tp
        n,m = X.shape
        Dparts = self.split(X,Y,T)
        
        def obji(Lv,i):
            L = np.diag(Lv)
            return self.objective(i,L)
        
        def obj(Lv):
            s = 0
            for i in range(0,parts):
                s += obji(Lv,i)
            return s
        
        if self.discrete:
            result = opt.minimize(obj,np.diag(self.L),method=method) #, options={'maxiter': 200}
            result.x = np.diag(result.x)
            self.L = result.x
        else:
            result = opt.minimize(obj,self.L.flatten(),method=method) #, options={'maxiter': 200}
            result.x = np.reshape(result.x,(m,m))
            self.L = result.x
        return self.L
    
    
    def CATE(self,Xtest1,Ytest1,Ttest1,L,K=5,is_y_continuous = True):
#        num_cores = multiprocessing.cpu_count()
        global Xtest, Ytest, Ttest, DpartsTest, dist_mat_test
        Xtest = Xtest1
        Ytest = Ytest1
        Ttest = Ttest1
        ntest,mtest = Xtest.shape
        
        if len(Ttest.shape)==1:
            tp = 1
        else:
            tp = T.shape[1]
        parts = 2**tp
        
        DpartsTest = self.split(Xtest,Ytest,Ttest)
        for c in range(0,parts):
            Xc, Yc, Tc = DpartsTest[c]
            dist_mat_test[c] = calc_pairwise_d(Xtest,Xc,L,test=True)
            
        ntest = len(Xtest)
        k = K
        t_hat = []
        nargs = [(i,L,k) for i in range(0,ntest) ]
        d = list(Parallel(n_jobs=(1+(ntest//50)), prefer="threads")(delayed(makeMatchedGroup)(args) for args in nargs))
        
        nargs1 = [ (i,d[i],tp,is_y_continuous) for i in range(0,ntest) ]
        t_hat = list(Parallel(n_jobs=(1+(ntest//50)), prefer="threads")(delayed(ITE)(args1) for args1 in nargs1))

        return d,t_hat
    
    def ATE(self,Xtest,X,Y,T,fL,is_y_continuous = True):
        d,t_hat = self.CATE(Xtest,X,Y,T,fL,is_y_continuous)
        return np.average(t_hat,axis=0)