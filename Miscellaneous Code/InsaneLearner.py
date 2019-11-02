# -*- coding: utf-8 -*-
"""
Created on Thu May 03 13:25:19 2018

@author: Harsh
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

class InsaneLearner:
    
    def __init__(self,n_estimators=100):
        self.rfC = RandomForestRegressor(n_estimators=n_estimators)
        self.rfT = RandomForestRegressor(n_estimators=n_estimators)
        
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
    
    def insaneLearning(self,X,Y,T):
        (control,treatment)=self.split(X,Y,T)
        Xc,Yc,Tc = control
        Xt,Yt,Tt = treatment
        self.rfC = self.rfC.fit(Xc,Yc)
        self.rfT = self.rfT.fit(Xt,Yt)
        
    def ITE(self,x):
        yC = self.rfC.predict(x)
        yT = self.rfT.predict(x)
        return yT - yC
    
    def CATE(self,X):
        YC = self.rfC.predict(X)
        YT = self.rfT.predict(X)
        return np.array(YT) - np.array(YC)
    