# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:13:53 2018

@author: Harsh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 01:19:15 2018

@author: Harsh
"""
import numpy as np
import scipy.optimize as opt
from sklearn import cluster as cluster
from matplotlib.pyplot import *
import time
import pandas as pd
import data_generation as dg
from pathos.multiprocessing import ProcessingPool as Pool
import os
#import compas
#import FLAMEbit

def d2(xi,xj,L,dcov=[]):
    #np.eye(number_of_covariates + ( len(dcov)*(len(dcov)-1) ) )
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

def d1(xi,xj,L,dcov=[]):
    return np.dot(xi-xj,np.dot(L,xi-xj))
    
    
class distance_metric_learning:
    def __init__(self,number_of_covariates,diag=True,dcov=[],K=10,L=None,distance=d):
        self.discrete = diag
        if L is None:
            self.L = np.eye(number_of_covariates)
        else:
            self.L = L
        self.K = K
        self.num_covariates = number_of_covariates
        self.dcov = dcov
        self.distance = lambda xi,xj,Lij: distance(xi,xj,Lij,self.dcov)
               
#    def distance(self,xi,xj,Li,Lj=None):
#        return np.dot(xi-xj,np.dot(Li,xi-xj))

    
    def kneighbors(self,i,X,Y,T,L):
        n,m = X.shape
        if n<self.K:
            return [],[]
        xi = X[i,:]
        yi = Y[i]
        kn = [ X[j,:] for j in range(0,self.K) ]
        ykn = [ Y[j] for j in range(0,self.K) ]
        dkn = [self.distance(xi,xj,L) for xj in kn]
        maxj = np.argmax( dkn )
        maxdkn = np.max( dkn )
        for j in range(self.K,n):
            dij = self.distance(xi,X[j,:],L)
            if dij < maxdkn:
                kn[maxj] = X[j,:]
                ykn[maxj] = Y[j]
                dkn[maxj] = dij
                maxj = np.argmax( dkn )
                maxdkn = np.max( dkn )
        return kn, ykn
    
    def objective(self,X,Y,T,L):
        n,m = X.shape
        if n<self.K:
            return 0
        obj = 0
        for i in range(0,n):
            kni, ykni = self.kneighbors(i,X,Y,T,L)
            if len(ykni)!=0:
                w = np.array(list(map( lambda x: np.exp(-1*(0)*self.distance(X[i,:],x,L)), kni )))
                obj += np.square(Y[i] - (1.0/sum(w))*np.sum(w*ykni))
        diff = np.linalg.norm(L,ord='nuc')
        c_coeff = 0.0001
        cost = c_coeff*diff
        return obj + cost
    
    def softobjective(self,X,Y,T,L,regfunc=None,regcost=0):
        n = len(Y)
        D = np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                D[i,j] = np.exp( -10 * self.distance(X[i],X[j],L) )
        Dd = np.diag(np.diag(D))
        gamma = np.dot(D-Dd,Y)
        delta = np.dot(D-Dd,np.ones((n,)))
        vec = Y - (gamma/delta)
        cost = np.dot(vec,vec)
        if regfunc is not None:
            cost = cost + regcost*regfunc(L)
        return cost
    
    def numerical_gradient(self,X,Y,T,L,obj,eps=0.005,searchspace=None):
        if searchspace is None:
            searchspace = np.arange(0,self.num_covariates)
        #print(L.shape)
        p,_ = L.shape
        grad = np.zeros((p,))
        for i in searchspace:
            z = np.zeros((p,p))
            z[i,i] = eps
            Lprime1 = obj(X,Y,T,np.copy(L) + np.copy(z))
            Lprime2 = obj(X,Y,T,np.copy(L) - np.copy(z))
            #print((Lprime1,Lprime2))
            dobj = Lprime1 - Lprime2
            grad[i] = dobj/(2*eps)
        return grad
        
    
    def split(self,X,Y,T):
        n,m = X.shape
        Xc,Tc,Yc = [],[],[]
        Xt,Tt,Yt = [],[],[]
        for i in range(0,n):
            if T[i] == 0:
                Xc.append(X[i,:])
                Yc.append(Y[i])
                Tc.append(T[i])
            elif T[i] == 1:
                Xt.append(X[i,:])
                Yt.append(Y[i])
                Tt.append(T[i])
        #print Xc is None,Yc is None,Tc is None,Xt is None,Yt is None,Tt is None
        return (np.array(Xc),np.array(Yc),np.array(Tc)),(np.array(Xt),np.array(Yt),np.array(Tt))
    
    def mkbatch(self,X,Y,T,numbatches):
        try:
            Dc, Dt = self.split(X,Y,T)
        except:
            print(len(Y))
            raise
        def mkbhelper(dc,dt,numbatch):
            Xc,Yc,Tc = dc
            Xt,Yt,Tt = dt
            nc,mc = Xc.shape
            nt,mt = Xt.shape
            bsc = int(nc/(numbatch))
            bst = int(nt/(numbatch))
            batches = []
            i = 0
            while i < nc:
                j = 0
                Xb,Yb,Tb =[],[],[]
                while j < bsc:
                    Xb.append(Xc[i,:])
                    Yb.append(Yc[i])
                    Tb.append(Tc[i])
                    i += 1
                    j += 1
                    if i>= nc:
                        break
                batches.append( ( Xb ,  Yb ,  Tb  ) )
            i = 0
            k = 0
            while i < nt:
                j = 0
                Xb,Yb,Tb = batches[k]
                while j < bst:
                    Xb.append(Xt[i,:])
                    Yb.append(Yt[i])
                    Tb.append(Tt[i])
                    i += 1
                    j += 1
                    if i>= nt:
                        break
                batches[k] = np.array(Xb),np.array(Yb),np.array(Tb)
                k += 1
                if k>= len(batches):
                    break
            return batches
        return mkbhelper(Dc,Dt,numbatches)
    
    def gradient(self,X,Y,T,L):
        n,p = X.shape
        deltaCap = np.zeros((n,n))
        delta = np.zeros((n,))
        for i in range(0,n):
            for j in range(0,n):
                deltaCap[i,j] = np.exp(-1*self.distance(X[i,:],X[j,:],L))
        for i in range(0,n):
            delta[i] = np.sum(deltaCap[i,:]) - deltaCap[i,i]
        grad = np.zeros((p,))
        for k in range(0,p):
            grad[k] = np.sum( [ 2*( Y[i] - ( np.sum([deltaCap[i,j]*Y[j] for j in range(0,n) if j!= i ])/delta[i] ) )* np.sum( [ ( (deltaCap[i,j]*np.square(X[i,k]-X[j,k])/delta[i]) - (deltaCap[i,j]*np.sum( [ (deltaCap[i,j1]*np.square(X[i,k]-X[j1,k])) for j1 in range(0,n) if j1 != i ] )/ np.square(delta[i]) ) )*Y[j] for j in range(0,n) if j!= i] ) for i in range(0,n)] )
        return grad
        
    def softminimizeGD(self,X,Y,T,L,regfunc = lambda x: np.linalg.norm(x,ord='fro'),regcost=100,searchspace=None,itr=5):
        if searchspace is None:
            searchspace = np.arange(0,self.num_covariates)
        alpha = 0.05
        #L = L/np.max(np.abs(L))
        loss = []
        obj = lambda x,y,t,l: self.softobjective(x,y,t,l,regfunc,regcost)
        for i in range(0,itr):
            print(i)
            o1 = obj(X,Y,T,L)
            loss.append(o1)
            g = self.numerical_gradient(X,Y,T,L,obj,searchspace=searchspace)
            if np.isnan(g).any():
                break
            gL = np.diag(g)
            # print(np.max(gL))
            Lold = np.copy(L)
            L = L - alpha*gL
            #L = L/np.max(np.abs(L))
            o2 = obj(X,Y,T,L)
            if o1 < o2:
                L = np.copy(Lold) 
                alpha = alpha/2.0#+ np.diag(np.random.normal(0,0.005,len(g)))
            #print(g)
            #print((o1,o2))
        return L,loss
    
    def optimizeGD(self,X,Y,T,regfunc = lambda x: np.linalg.norm(x,ord='fro'),regcost=0,numbatch=1,searchspace=None):
        indices = np.arange(0,len(Y))
        np.random.shuffle(indices)
        X,Y,T = X[indices],Y[indices],T[indices]
        if searchspace is None:
            searchspace = np.arange(0,self.num_covariates)
        batches = self.mkbatch(X,Y,T,numbatch)
        lossa = []
        for itr in range(0,1):
            #print(itr)
            for batch in batches:
                Xb,Yb,Tb = batch
#                Xtr,Ytr,Ttr = Xb[:3*len(Yb)//4],Yb[:3*len(Yb)//4],Tb[:3*len(Yb)//4]
#                Xv,Yv,Tv = Xb[3*len(Yb)//4:],Yb[3*len(Yb)//4:],Tb[3*len(Yb)//4:]
                n,m = Xb.shape
                (control,treatment)=self.split(Xb,Yb,Tb)
                Xc,Yc,Tc = control
                Xt,Yt,Tt = treatment
                L,lossc = self.softminimizeGD(Xc,Yc,Tc,self.L,regfunc=regfunc,regcost=regcost,searchspace=searchspace)
                self.L = L
                lossbc = np.array(lossc)
                #print(np.mean(loss))
                L,losst = self.softminimizeGD(Xt,Yt,Tt,self.L,regfunc=regfunc,regcost=regcost,searchspace=searchspace)
                self.L = L
                lossbt = np.array(losst)
                lossb = (len(Yc)*lossbc + len(Yt)*lossbt)/n
                lossa.append(lossb)
                #print(np.mean(loss))
        return self.L, lossa
                
    
    def optimize(self,X,Y,T,soft=False,numbatch=1,method='COBYLA'):
#        X,Y,T = self.normalize(Xu,Yu,Tu)
        indices = np.arange(0,len(Y))
        np.random.shuffle(indices)
        X,Y,T = X[indices],Y[indices],T[indices]
        batches = self.mkbatch(X,Y,T,numbatch)
        for itr in range(0,1):
            for batch in batches:
                Xb,Yb,Tb = batch
                n,m = Xb.shape
                (control,treatment)=self.split(Xb,Yb,Tb)
                Xc,Yc,Tc = control
                Xt,Yt,Tt = treatment
                def objC(Lv):
                    if self.discrete:
                        L = np.diag(Lv)
                    else:
                        L = np.reshape(Lv,(m,m))
                    if soft == True:
                        return self.softobjective(Xc,Yc,Tc,L)
                    else:
                        return self.objective(Xc,Yc,Tc,L)
                def objT(Lv):
                    if self.discrete:
                        L = np.diag(Lv)
                    else:
                        L = np.reshape(Lv,(m,m))
                    if soft == True:
                        return self.softobjective(Xt,Yt,Tt,L)
                    else:
                        return self.objective(Xt,Yt,Tt,L)
                if self.discrete:
                    resultC = opt.minimize(objC,np.diag(self.L),method=method) #, options={'maxiter': 200}
                    resultC.x = np.diag(resultC.x)
                    self.L = resultC.x
                    resultT = opt.minimize(objT,np.diag(self.L),method=method) #, options={'maxiter': 200}
                    resultT.x = np.diag(resultT.x)
                    self.L = resultT.x
                else:
                    resultC = opt.minimize(objC,self.L.flatten(),method=method) #, options={'maxiter': 200}
                    resultC.x = np.reshape(resultC.x,(m,m))
                    self.L = resultC.x
                    resultT = opt.minimize(objT,self.L.flatten(),method=method) #, options={'maxiter': 200}
                    resultT.x = np.reshape(resultT.x,(m,m))
                    self.L = resultT.x
        return resultC,resultT
    
    def optimize_parallel(self,X,Y,T,iterations,numbatch):
        def ohp(tupl):
            X,Y,T,Lc,Lt = tupl
            soft=False
            method='COBYLA' 
            Xb,Yb,Tb = X,Y,T
            n,m = Xb.shape
            (control,treatment)=self.split(Xb,Yb,Tb)
            Xc,Yc,Tc = control
            Xt,Yt,Tt = treatment
            def objC(Lv):
                if self.discrete:
                    L = np.diag(Lv)
                else:
                    L = np.reshape(Lv,(m,m))
                if soft == True:
                    return self.softobjective(Xc,Yc,Tc,L)
                else:
                    return self.objective(Xc,Yc,Tc,L)
            def objT(Lv):
                if self.discrete:
                    L = np.diag(Lv)
                else:
                    L = np.reshape(Lv,(m,m))
                if soft == True:
                    return self.softobjective(Xt,Yt,Tt,L)
                else:
                    return self.objective(Xt,Yt,Tt,L)
            if self.discrete:
                resultC = opt.minimize(objC,np.diag(Lc),method=method, options={'maxiter': 200})
                resultC.x = np.diag(resultC.x)
                resultT = opt.minimize(objT,np.diag(Lt),method=method, options={'maxiter': 200})
                resultT.x = np.diag(resultT.x)
            else:
                resultC = opt.minimize(objC,Lc.flatten(),method=method, options={'maxiter': 100})
                resultC.x = np.reshape(resultC.x,(m,m))
                resultT = opt.minimize(objT,Lc.flatten(),method=method, options={'maxiter': 100})
                resultT.x = np.reshape(resultT.x,(m,m))
            Lc = resultC.x
            Lt = resultT.x
            return (Lc, Lc)
        pool = Pool(processes=numbatch)
        batches = self.mkbatch(X,Y,T,numbatch)
        for itr in range(0,iterations):
            print('iteration', itr)
            Lcbatches = [ np.copy(self.L) for i in range(0,len(batches))]
            Ltbatches = [ np.copy(self.L) for i in range(0,len(batches))]
            tArray = [ (batches[i][0],batches[i][1],batches[i][2],Lcbatches[i],Ltbatches[i]) for i in range(0,len(batches))]
            LoutArray = pool.map(ohp,tArray)
            #pool.close()
            Lc = np.average([ np.array(LoutArray[i][0]) for i in range(0,len(batches))],axis=0)
            self.L = Lc
            self.L = Lc
        #pool.join()
        return Lc, Lc

    def findneighbors(self,x,X,Y,fL,k = 10,soft=False):
        n,m = X.shape
        Xk = X[0:k,:]
        Yk = Y[0:k]
        dk = [ self.distance(x,X[i,:],fL(x)) for i in range(0,min(k,n)) ]
        for i in range(k,n):
            maxi = np.argmax(dk)
            di = self.distance(x,X[i,:],fL(x)) 
            if di < dk[maxi]:
                Xk[maxi,:] = X[i,:]
                Yk[maxi] = Y[i]
                dk[maxi] = di
        return np.array(Xk),np.array(Yk)
        
    def nearestneighbormatching(self,X,Y,T,fLC,fLT=None):
        if not fLT is None:
            fLT = fLC
        (control,treatment)=self.split(X,Y,T)
        Xc,Yc,Tc = control
        Xt,Yt,Tt = treatment
        nc = len(Yc)
        nt = len(Yt)
        k = 5
        d = {}
        dc = {}
        for i in range(0,nc):
            neighborsX, neighborsY = self.findneighbors(Xc[i,:],Xt,Yt,fLT,k)
            dc[i] = ((neighborsX,Xc[i,:]), (neighborsY,Yc[i]))
        d['c'] = (dc,1)
        dt = {}
        for i in range(0,nt):
            neighborsX, neighborsY = self.findneighbors(Xt[i,:],Xc,Yc,fLC,k)
            dt[i] = ((neighborsX,Xt[i,:]), (neighborsY,Yt[i]))
        d['t'] = (dt,-1)
        return d
    
    def CATE(self,Xtest,X,Y,T,fLC,fLT=None,beta=None):
        if not fLT is None:
            fLT = fLC
        (control,treatment)=self.split(X,Y,T)
        Xc,Yc,Tc = control
        Xt,Yt,Tt = treatment
        ntest = len(Xtest)
        k = self.K
        d = {}
        t_hat = []
        for i in range(0,ntest):
            neighborsXc, neighborsYc = self.findneighbors(Xtest[i,:],Xc,Yc,fLC,k)
            neighborsXt, neighborsYt = self.findneighbors(Xtest[i,:],Xt,Yt,fLC,k)
            d[i] = ((neighborsXc,neighborsXt), (neighborsYc,neighborsYt), Xtest[i,:])
        for k,v in d.items():
            t_hat.append(np.average(v[1][1]) - np.average(v[1][0]))
        return d,t_hat
        
    def ATE(self,D):
        CTE = 0
        count = 0
        for key,g in D.items():
            d,t = g
            for k,v in d.items():
                xs, ys = v
                TE = np.average(ys[0]) - ys[1]
                CTE += t*TE
                count += 1
        return CTE/float(count)
    
    def printNdigits(self,x):
        return float("{0:.4f}".format(x))
    
    def printD(self,D,filename='matchedGroups'):
        fd = open(filename,'w')
        for key,g in D.items():
            print(key, file=fd)
            x,y,q = g
            xc,xt = x
            yc,yt = y
            np.savetxt(fd,np.array(q),delimiter=',')
            np.savetxt(fd,np.array(xc),delimiter=',')
            np.savetxt(fd,np.array(yc),delimiter=',')
            np.savetxt(fd,np.array(xt),delimiter=',')
            np.savetxt(fd,np.array(yt),delimiter=',')
        return 0
    
    def cross_validate(self,Xval,Yval,Tval):
        return 0
        