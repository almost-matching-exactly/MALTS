# -*- coding: utf-8 -*-
"""
Created on Thu May 03 13:25:19 2018

@author: Harsh
"""

import numpy as np

class InsaneLearner:
    
    def __init__(self,n_estimators=100):
        rfC = RandomForestRegressor(n_estimators=100)
        rfT = RandomForestRegressor(n_estimators=100)
    
    def insaneLearning(self,X,Y,T):
        from sklearn.ensemble import RandomForestRegressor
        rfC = RandomForestRegressor(n_estimators=100)
        rfT = RandomForestRegressor(n_estimators=100)
        (control,treatment)=self.split(X,Y,T)
        Xc,Yc,Tc = control
        Xt,Yt,Tt = treatment
        rfC = rfC.fit(Xc,Yc)
        rfT = rfT.fit(Xt,Yt)
        retu