#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:01:15 2020

@author: harspari
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR

class neuromalts:
    def __init__( self, outcome, treatment, data, discrete, model='svm' ):
        self.outcome = outcome
        self.treatment = treatment
        self.discrete = discrete
        self.continuous = list(set(data.columns).difference(set([outcome]+[treatment]+discrete)))
        self.Y = data[outcome]
        self.T = data[treatment]
        self.X = data[self.continuous+self.discrete]
        if model == 'svm':
            self.dmap = SVR(kernel='rbf')
        
    def 