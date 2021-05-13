#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:01:21 2020

@author: harshparikh
"""

import numpy as np
import sklearn.ensemble as ensemble
import pandas as pd

from sklearn.model_selection import StratifiedKFold

class prognostic:
    def __init__(self,Y,T,df,binary=False):
        self.Y = Y
        self.T = T
        self.df = df
        self.cov = list(set(df.columns).difference(set([Y]+[T])))
        self.df_c = df.loc[df[T]==0]
        self.df_t = df.loc[df[T]==1]
        self.Xc, self.Yc =self.df_c[self.cov], self.df_c[Y]
        self.Xt, self.Yt = self.df_t[self.cov], self.df_t[Y]
        self.hc = ensemble.RandomForestRegressor(n_estimators=100).fit(self.Xc,self.Yc)
        self.ht = ensemble.RandomForestRegressor(n_estimators=100).fit(self.Xt,self.Yt)
    
    def get_matched_group(self,df_est,k=1):
        df_mg = pd.DataFrame(columns=['Yc','Yt','T','CATE'])
        df_e_c = df_est.loc[df_est[self.T]==0]
        df_e_t = df_est.loc[df_est[self.T]==1]
        Xec, Yec = df_e_c[self.cov].to_numpy(), df_e_c[self.Y].to_numpy()
        Xet, Yet = df_e_t[self.cov].to_numpy(), df_e_t[self.Y].to_numpy()
        hatYcc = self.hc.predict(Xec)
        hatYct = self.hc.predict(Xet)
        hatYtc = self.ht.predict(Xec)
        hatYtt = self.ht.predict(Xet)
        for i in range(0,len(hatYct)):
            ps = hatYct[i]
            dis = np.abs(hatYcc-ps)
            idx = np.argpartition(dis, k)
            df_temp = pd.DataFrame()
            df_temp['Yc'] = [np.mean(Yec[idx[:k]])]
            df_temp['Yt'] = [Yet[i]]
            df_temp['T'] = [1]
            df_temp['CATE'] = [Yet[i] - np.mean(Yec[idx[:k]])]
            df_temp = df_temp.rename(index={0:df_e_t.index[i]})
            df_mg = df_mg.append(df_temp)
        for i in range(0,len(hatYtc)):
            ps = hatYtc[i]
            dis = np.abs(hatYtt-ps)
            idx = np.argpartition(dis, k)
            df_temp = pd.DataFrame()
            df_temp['Yc'] = [Yec[i]]
            df_temp['Yt'] = [np.mean(Yet[idx[:k]])]
            df_temp['T'] = [0]
            df_temp['CATE'] = [np.mean(Yet[idx[:k]]) - Yec[i]]
            df_temp = df_temp.rename(index={0:df_e_c.index[i]})
            df_mg = df_mg.append(df_temp)
        return df_mg
    
def prognostic_cv(outcome, treatment, data,n_splits=5):
    np.random.seed(0)
    skf = StratifiedKFold(n_splits=n_splits)
    gen_skf = skf.split(data,data[treatment])
    cate_est = pd.DataFrame()
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        prog = prognostic(outcome,treatment,df_train)
        prog_mg = prog.get_matched_group(df_est) 
        cate_est_i = pd.DataFrame(prog_mg['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est
        

            