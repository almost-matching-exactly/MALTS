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
#import neuralnet
from sklearn import cluster as cluster
from matplotlib.pyplot import *
import time
import pandas as pd
import data_generation as dg
from pathos.multiprocessing import ProcessingPool as Pool
import os
#import compas
#import FLAMEbit

class distance_metric_learning:
    def __init__(self,number_of_covariates,discrete=False):
        self.discrete = discrete
        self.Lc = np.eye(number_of_covariates)
        self.Lt = np.eye(number_of_covariates)
        self.K = 10
        #self.nnet = neuralnet.NeuralNet(number_of_covariates,(number_of_covariates*number_of_covariates),[int(0.5*number_of_covariates)])
        
    def __distance(self,xi,xj,Li,Lj=None,order=2):
        if self.discrete == False:
            if Lj is None:
                Lj = Li
            return np.linalg.norm(np.dot(Li,xi)-np.dot(Lj,xj),ord=order)
        else:
            if Lj is None:
                Lj = Li
            return np.linalg.norm(np.dot(Li,xi)-np.dot(Lj,xj),ord=order)
    
    def __kneighbors(self,i,X,Y,T,L):
        n,m = X.shape
        xi = X[i,:]
        yi = Y[i]
        kn = [ X[j,:] for j in range(0,self.K) ]
        ykn = [ Y[j] for j in range(0,self.K) ]
        dkn = [self.__distance(xi,xj,L) for xj in kn]
        maxj = np.argmax( dkn )
        maxdkn = np.max( dkn )
        for j in range(self.K,n):
            dij = self.__distance(xi,X[j,:],L)
            if dij < maxdkn:
                kn[maxj] = X[j,:]
                ykn[maxj] = Y[j]
                dkn[maxj] = dij
                maxj = np.argmax( dkn )
                maxdkn = np.max( dkn )
        return kn, ykn
    
    def __objective(self,X,Y,T,L):
        n,m = X.shape
        obj = 0
        for i in range(0,n):
            kni, ykni = self.__kneighbors(i,X,Y,T,L)
            obj += np.square(Y[i] - (1.0/self.K)*np.sum(ykni))
        return obj
    
    def __softobjective(self,X,Y,T,L):
        n,m = X.shape
        obj = 0
        for i in range(0,n):
            obj += np.square(Y[i] - (np.sum([ np.exp(-0.01*self.__distance(X[i],X[j],L)) * Y[j] for j in range(0,n) ])/np.sum([ np.exp(-0.01*self.__distance(X[i],X[j],L)) for j in range(0,n) ])))
        return obj
    
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
    
    def __mkbatch(self,X,Y,T,numbatches):
        n,m = X.shape
        bs = int(n/(numbatches))
        batches = []
        i = 0
        while i < n:
            j = 0
            Xb,Yb,Tb =[],[],[]
            while j < bs:
                Xb.append(X[i,:])
                Yb.append(Y[i])
                Tb.append(T[i])
                i += 1
                j += 1
                if i>= n:
                    break
            batches.append( ( np.array( Xb ), np.array( Yb ), np.array( Tb ) ) )
        return batches
        
        
    
    def optimize(self,X,Y,T,soft=True,numBatch=10,method='COBYLA'):
        batches = self.__mkbatch(X,Y,T,numBatch)
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
                        return self.__softobjective(Xc,Yc,Tc,L)
                    else:
                        return self.__objective(Xc,Yc,Tc,L)
                def objT(Lv):
                    if self.discrete:
                        L = np.diag(Lv)
                    else:
                        L = np.reshape(Lv,(m,m))
                    if soft == True:
                        return self.__softobjective(Xt,Yt,Tt,L)
                    else:
                        return self.__objective(Xt,Yt,Tt,L)
                if self.discrete:
                    resultC = opt.minimize(objC,np.diag(self.Lc),method=method) #, options={'maxiter': 200}
                    resultC.x = np.diag(resultC.x)
                    resultT = opt.minimize(objT,np.diag(self.Lt),method=method) #, options={'maxiter': 200}
                    resultT.x = np.diag(resultT.x)
                else:
                    resultC = opt.minimize(objC,self.Lc.flatten(),method=method) #, options={'maxiter': 200}
                    resultC.x = np.reshape(resultC.x,(m,m))
                    resultT = opt.minimize(objT,self.Lt.flatten(),method=method) #, options={'maxiter': 200}
                    resultT.x = np.reshape(resultT.x,(m,m))
                self.Lc = resultC.x
                self.Lt = resultT.x
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
                    return self.__softobjective(Xc,Yc,Tc,L)
                else:
                    return self.__objective(Xc,Yc,Tc,L)
            def objT(Lv):
                if self.discrete:
                    L = np.diag(Lv)
                else:
                    L = np.reshape(Lv,(m,m))
                if soft == True:
                    return self.__softobjective(Xt,Yt,Tt,L)
                else:
                    return self.__objective(Xt,Yt,Tt,L)
            if self.discrete:
                resultC = opt.minimize(objC,np.diag(Lc),method=method, options={'maxiter': 200})
                resultC.x = np.diag(resultC.x)
                resultT = opt.minimize(objT,np.diag(Lt),method=method, options={'maxiter': 200})
                resultT.x = np.diag(resultT.x)
            else:
                resultC = opt.minimize(objC,Lc.flatten(),method=method, options={'maxiter': 100})
                resultC.x = np.reshape(resultC.x,(m,m))
                resultT = opt.minimize(objT,Lt.flatten(),method=method, options={'maxiter': 100})
                resultT.x = np.reshape(resultT.x,(m,m))
            Lc = resultC.x
            Lt = resultT.x
            return (Lc, Lt)
        pool = Pool(processes=numbatch)
        batches = self.__mkbatch(X,Y,T,numbatch)
        for itr in range(0,iterations):
            print(( 'iteration'+str( itr) ))
            Lcbatches = [ np.copy(self.Lc) for i in range(0,len(batches))]
            Ltbatches = [ np.copy(self.Lt) for i in range(0,len(batches))]
            tArray = [ (batches[i][0],batches[i][1],batches[i][2],Lcbatches[i],Ltbatches[i]) for i in range(0,len(batches))]
            LoutArray = pool.map(ohp,tArray)
            #pool.close()
            Lc = np.average([ np.array(LoutArray[i][0]) for i in range(0,len(batches))],axis=0)
            Lt = np.average([ np.array(LoutArray[i][1]) for i in range(0,len(batches))],axis=0)
            self.Lc = Lc
            self.Lt = Lt
        #pool.join()
        return Lc, Lt
    
#    def optimizeneuralnet(self,X,Y,T,iterations=5,numbatch=10):
#        nnet=self.nnet
#        for itr in range(0,iterations):
#            batches = self.__mkbatch(X,Y,T,numbatch)
#            for batch in batches:
#                Xb,Yb,Tb = batch
#                n,m = Xb.shape
#                (control,treatment)=self.split(Xb,Yb,Tb)
#                Xc,Yc,Tc = control
#                Xt,Yt,Tt = treatment
#                def objectivenn(Xb,Yb,Tb,i,Li):
#                    #L2i = np.diag(Li)
#                    L2i = Li.reshape((m,m))
#                    if Tb[i]==0:
#                        nc,mc = Xc.shape
#                        return np.square(Yb[i] - (np.sum([ np.exp(-0.01*self.__distance(Xb[i,:],Xc[j,:],L2i,np.reshape(nnet.predict(Xc[j,:]),(m,m)))) * Yc[j] for j in range(0,nc) ])/np.sum([ np.exp(-0.01*self.__distance(Xb[i,:],Xc[j,:],L2i,np.reshape(nnet.predict(Xc[j,:]),(m,m)))) for j in range(0,nc) ])))
#                    elif Tb[i]==1:
#                        nt,mt = Xt.shape
#                        return np.square(Yb[i] - (np.sum([ np.exp(-0.01*self.__distance(Xb[i,:],Xt[j,:],L2i,np.reshape(nnet.predict(Xt[j,:]),(m,m)))) * Yt[j] for j in range(0,nt) ])/np.sum([ np.exp(-0.01*self.__distance(Xb[i,:],Xt[j,:],L2i,np.reshape(nnet.predict(Xt[j,:]),(m,m)))) for j in range(0,nt) ])))
#                for i in range(0,n):
#                    Lhat = np.array(nnet.predict(Xb[i,:]))
#                    dnnet = nnet.errorPropogate(Lhat,f=(lambda L: objectivenn(Xb,Yb,Tb,i,L)))
#                    nnet.updateNN(0.5/np.sqrt(itr+1))
#        return nnet

    def findneighbors(self,x,X,Y,fL,k = 10,soft=False):
        n,m = X.shape
        if soft==True:
            retX = [(np.sum([ np.exp(-1*self.__distance(x,X[j,:],fL(x),fL(X[j,:]))) * X[j,:] for j in range(0,n) ])/np.sum([ np.exp(-1*self.__distance(x,X[j,:],fL(x),fL(X[j,:]))) for j in range(0,n) ]))]
            retY = [(np.sum([ np.exp(-1*self.__distance(x,X[j,:],fL(x),fL(X[j,:]))) * Y[j] for j in range(0,n) ])/np.sum([ np.exp(-1*self.__distance(x,X[j,:],fL(x),fL(X[j,:]))) for j in range(0,n) ]))]
            return retX,retY
        else:
            Xk = X[0:k,:]
            Yk = Y[0:k]
            dk = [ self.__distance(x,X[i,:],fL(x)) for i in range(0,k) ]
            for i in range(k,n):
                maxi = np.argmax(dk)
                di = self.__distance(x,X[i,:],fL(x),fL(X[i,:])) 
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
    
    def CATE(self,Xtest, X,Y,T,fLC,fLT=None,beta=None):
        if not fLT is None:
            fLT = fLC
        (control,treatment)=self.split(X,Y,T)
        Xc,Yc,Tc = control
        Xt,Yt,Tt = treatment
        ntest = len(Xtest)
        k = 10
        d = {}
        t_hat = []
        for i in range(0,ntest):
            neighborsXc, neighborsYc = self.findneighbors(Xtest[i,:],Xc,Yc,fLC,k)
            neighborsXt, neighborsYt = self.findneighbors(Xtest[i,:],Xt,Yt,fLT,k)
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
            d,t = g
            for k,v in d.items():
                xs, ys = v
                print('===================================', file=fd)
                print((list(map(self.printNdigits,xs[1])),self.printNdigits(ys[1])), file=fd)
                print('-----------------------------------', file=fd)
                for i in range(0,len(xs[0])):
                    print(( list(map(self.printNdigits,xs[0][i])),self.printNdigits(ys[0][i])), file=fd)
                print('===================================', file=fd)
        return 0