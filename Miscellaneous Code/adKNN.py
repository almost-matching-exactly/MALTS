# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 01:19:15 2018

@author: Harsh
"""
import numpy as np
import scipy.optimize as opt
import neuralnet
from sklearn import cluster as cluster
from matplotlib.pyplot import *
import time
#import compas
#import FLAMEbit

K=10

def distance(xi,xj,Li,Lj=None):
    if Lj is None:
        return np.linalg.norm(np.dot(Li,(xi-xj)))
    else:
        return np.linalg.norm(np.dot(Li,xi)-np.dot(Lj,xj))

def kneighbors(i,X,Y,T,L):
    n,m = X.shape
    xi = X[i,:]
    yi = Y[i]
    kn = [ X[j,:] for j in range(0,K) ]
    ykn = [ Y[j] for j in range(0,K) ]
    dkn = [distance(xi,xj,L) for xj in kn]
    maxj = np.argmax( dkn )
    maxdkn = np.max( dkn )
    for j in range(K,n):
        dij = distance(xi,X[j,:],L)
        if dij < maxdkn:
            kn[maxj] = X[j,:]
            ykn[maxj] = Y[j]
            dkn[maxj] = dij
            maxj = np.argmax( dkn )
            maxdkn = np.max( dkn )
    return kn, ykn

def objective(X,Y,T,L):
    n,m = X.shape
    obj = 0
    for i in range(0,n):
        kni, ykni = kneighbors(i,X,Y,T,L)
        obj += np.square(Y[i] - (1.0/K)*np.sum(ykni))
    return obj

def softobjective(X,Y,T,L):
    n,m = X.shape
    obj = 0
    for i in range(0,n):
        obj += np.square(Y[i] - (np.sum([ np.exp(-1*distance(X[i],X[j],L)) * Y[j] for j in range(0,n) ])/np.sum([ np.exp(-1*distance(X[i],X[j],L)) for j in range(0,n) ])))
    return obj

def split(X,Y,T):
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
    return (np.array(Xc),np.array(Yc),np.array(Tc)),(np.array(Xt),np.array(Yt),np.array(Tt))

def mkbatch(X,Y,T,numbatches):
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

def optimize(objective,X,Y,T,L0):
    (control,treatment)=split(X,Y,T)
    Xc,Yc,Tc = control
    Xt,Yt,Tt = treatment
    def obj(Lv):
        n2 = len(Lv)
        L = Lv.reshape((int(np.sqrt(n2)),int(np.sqrt(n2))))
        return objective(Xc,Yc,Tc,L)+objective(Xt,Yt,Tt,L)
    result = opt.minimize(obj,L0.flatten(),method='COBYLA', options={'maxiter': 500})
    Lstar = result.x
    n2 = len(Lstar)
    result.x = Lstar.reshape((int(np.sqrt(n2)),int(np.sqrt(n2))))
    return result
#
#def optimizeGreedy(objective,X,Y,T,L0):
#    result = optimize(objective,X,Y,T,L0)
#    L = result.x
#    Xb,Yb,Tb = findBadPerformers(X,Y,T,L,eps)
#    L1 = optimize(objective,Xb,Yb,Tb,L)
#    return L,L1


def optimizeneuralnet(X,Y,T,nnet,iterations,numbatch=10):
    for itr in range(0,iterations):
        batches = mkbatch(X,Y,T,numbatch)
        for batch in batches:
            Xb,Yb,Tb = batch
            n,m = Xb.shape
            (control,treatment)=split(Xb,Yb,Tb)
            Xc,Yc,Tc = control
            Xt,Yt,Tt = treatment
            def objective(Xb,Yb,Tb,i,Li):
                L2i = np.diag(Li)
                #n2 = len(Li)
                #L2i = Li.reshape((int(np.sqrt(n2)),int(np.sqrt(n2))))
                if Tb[i]==0:
                    nc,mc = Xc.shape
                    return np.square(Yb[i] - (np.sum([ np.exp(-1*distance(Xb[i,:],Xc[j,:],L2i,np.diag(nnet.predict(Xc[j,:])))) * Yc[j] for j in range(0,nc) ])/np.sum([ np.exp(-1*distance(Xb[i,:],Xc[j,:],L2i,np.diag(nnet.predict(Xc[j,:])))) for j in range(0,nc) ])))
                elif Tb[i]==1:
                    nt,mt = Xt.shape
                    return np.square(Yb[i] - (np.sum([ np.exp(-1*distance(Xb[i,:],Xt[j,:],L2i,np.diag(nnet.predict(Xt[j,:])))) * Yt[j] for j in range(0,nt) ])/np.sum([ np.exp(-1*distance(Xb[i,:],Xt[j,:],L2i,np.diag(nnet.predict(Xt[j,:])))) for j in range(0,nt) ])))
            for i in range(0,n):
                Lhat = np.array(nnet.predict(Xb[i,:]))
                dnnet = nnet.errorPropogate(Lhat,f=(lambda L: objective(Xb,Yb,Tb,i,L)))
                nnet.updateNN(0.05)
    return nnet

#def generateDataNonLinear(numFeatures,percentSignificant,numEx):
#    threshold = np.random.uniform(0.35,0.65)
#    X,Y,T = [],[],[]
#    nsf = int(numFeatures*percentSignificant)
#    nisf = numFeatures - nsf
#    beta = []
#    for i in range(0,nsf):
#        beta.append(np.random.uniform(5,10))
#    for i in range(0,nisf):
#        beta.append(np.random.uniform(0,1))
#    betat = 100
#    return 0
    
def generateData(numFeatures,percentSignificant,numEx):
    threshold = np.random.uniform(0.35,0.65)
    X,Y,T = [],[],[]
    nsf = int(numFeatures*percentSignificant)
    nisf = numFeatures - nsf
    beta = []
    for i in range(0,nsf):
        beta.append(np.random.uniform(5,10))
    for i in range(0,nisf):
        beta.append(np.random.uniform(0,1))
    betat = 100
    for i in range(0,numEx):
        xi = np.random.lognormal(0,1,numFeatures)
        if np.random.uniform(0,1) > threshold:
        #if ((0.3+0.2*(beta[0]*xi[0] + beta[-1]*xi[-1])) > 0.5):
            ti = 1
        else:
            ti = 0
        yi = np.square(np.dot(beta,xi)) + betat*ti + np.random.normal(0,0.25)
        X.append(xi)
        Y.append(yi)
        T.append(ti)
    return np.array(X),np.array(Y),np.array(T)

def discretize(X,L):
    n,m = L.shape
    for i in range(0,n):
        X[:,i] = list(map( int, X[:,i]*L[i,i] ))
    return X

def cluster(X,Y,T,L):
    dbs = cluster.DBSCAN( eps=10, metric=lambda x,y : distance(x,y,L) )
    clust = dbs.fit_predict(X)
    return clust

def findneighbors(x,X,Y,fL,k):
    n,m = X.shape
    Xk = X[0:k,:]
    Yk = Y[0:k]
    dk = [ distance(x,X[i,:],fL(x),fL(X[i,:])) for i in range(0,k) ]
    for i in range(k,n):
        maxi = np.argmax(dk)
        di = distance(x,X[i,:],fL(x),fL(X[i,:])) 
        if di < dk[maxi]:
            Xk[maxi,:] = X[i,:]
            Yk[maxi] = Y[i]
            dk[maxi] = di
    return np.array(Xk),np.array(Yk)
        
def nearestneighbormatching(X,Y,T,fL):
    (control,treatment)=split(X,Y,T)
    Xc,Yc,Tc = control
    Xt,Yt,Tt = treatment
    nc = len(Yc)
    nt = len(Yt)
    k = 5
    d = {}
    dc = {}
    for i in range(0,nc):
        neighborsX, neighborsY = findneighbors(Xc[i,:],Xt,Yt,fL,k)
        dc[i] = ((neighborsX,Xc[i,:]), (neighborsY,Yc[i]))
    d['c'] = (dc,1)
    dt = {}
    for i in range(0,nt):
        neighborsX, neighborsY = findneighbors(Xt[i,:],Xc,Yc,fL,k)
        dt[i] = ((neighborsX,Xt[i,:]), (neighborsY,Yt[i]))
    d['t'] = (dt,-1)
    return d
    
def ATE(D):
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

def printNdigits(x):
    return float("{0:.4f}".format(x))

def printD(D,filename='matchedGroups'):
    fd = open(filename,'w')
    for key,g in D.items():
        print(key, file=fd)
        d,t = g
        for k,v in d.items():
            xs, ys = v
            print('===================================', file=fd)
            print((list(map(printNdigits,xs[1])),printNdigits(ys[1])), file=fd)
            print('-----------------------------------', file=fd)
            for i in range(0,len(xs[0])):
                print(( list(map(printNdigits,xs[0][i])),printNdigits(ys[0][i])), file=fd)
            print('===================================', file=fd)
    return 0
            

currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('log'+currenttime+'.txt','w')
numExperiment = 1
numExample = 1000
numVariable = 5
ATEcobylaArray = []
ATEneuralnetArray = []
print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
for i in range(0,numExperiment):
    print(i)
    X,Y,T = generateData(numVariable,0.6,numExample)
    L0 =  np.eye(numVariable,dtype=float) #np.random.uniform(0,1,size=(10,10))
    n,m = X.shape
    
    #COBYLA Optimization
    Lstar = optimize(objective,X,Y,T,L0)
    Lstar = Lstar.x
    fLcobyla = lambda y: Lstar
    dcobyla = nearestneighbormatching(X,Y,T,fLcobyla)
    ATEcobyla = ATE(dcobyla)
    print('ATE COBYLA:- '+str(i)+' : '+str( ATEcobyla), file=log)
    ATEcobylaArray.append(ATEcobyla)
    #Xd = discretize(X,L2star)
    
    #Neural Network Optimization
    #L(x[i]) -> min((y[i]-(sum{j in ex}w(L)[i,j]y[j])/(sum{j in ex}w(L)[i,j]y[j]))^2)
    iterations = 5
    nnet = neuralnet.NeuralNet(m,m,[int(m)])
    Lnnstar = optimizeneuralnet(X,Y,T,nnet,iterations)
    fLneuralnet = lambda x: np.diag(nnet.predict(x))
    dneuralnet = nearestneighbormatching(X,Y,T,fLneuralnet)
    ATEneuralnet = ATE(dneuralnet)
    print('ATE Neural Net:- '+str(i)+' : '+str( ATEneuralnet), file=log)
    ATEneuralnetArray.append(ATEneuralnet)
    
print(str(('Avg ATE COBYLA: ',np.average(ATEcobylaArray),' variance: ',np.var(ATEcobylaArray))), file=log)
print(str(('Avg ATE Neural Net: ',np.average(ATEneuralnetArray),' variance: ',np.var(ATEneuralnetArray))), file=log)

log.close()
#age = compas.age[:1000,:]
#decile_score = compas.decile_score[:1000]
#
#n,m = age.shape
#compas_nnet = neuralnet.NeuralNet(m,m*m,[m*4])
#compas_Lnnstar = optimizeneuralnet(age,decile_score,np.zeros((len(age),)),compas_nnet,1)
