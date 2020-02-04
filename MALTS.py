#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:43:45 2019

@author: harshparikh
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class malts:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,gamma=1,epsilon=600,k=10):
        self.C = C #coefficient to regularozation term
        self.gamma = gamma
        self.k = k
        self.epsilon = epsilon #lambda x: (1 + np.exp( - self.epsilon) )/(1+np.exp( self.gamma * (x - self.epsilon) ) )
        self.n, self.p = data.shape
        self.p = self.p - 2#shape of the data
        self.outcome = outcome
        self.treatment = treatment
        self.discrete = discrete
        self.continuous = set(data.columns).difference(set([outcome]+[treatment]+discrete))
#        Mc = np.ones((len(self.continuous),)) #initializing the stretch vector 
#        Md = np.ones((len(self.discrete),)) #initializing the stretch vector 
        #splitting the data into control and treated units
        self.df_T = data.loc[data[treatment]==1]
        self.df_C = data.loc[data[treatment]==0]
        #extracting relevant covariates (discrete,continuous) 
        #and outcome. Converting to numpy array.
        self.Xc_T = self.df_T[self.continuous].to_numpy()
        self.Xc_C = self.df_C[self.continuous].to_numpy()
        self.Xd_T = self.df_T[self.discrete].to_numpy()
        self.Xd_C = self.df_C[self.discrete].to_numpy()
        self.Y_T = self.df_T[self.outcome].to_numpy()
        self.Y_C = self.df_C[self.outcome].to_numpy()
        self.del2_Y_T = ((np.ones((len(self.Y_T),len(self.Y_T)))*self.Y_T).T - (np.ones((len(self.Y_T),len(self.Y_T)))*self.Y_T))**2
        self.del2_Y_C = ((np.ones((len(self.Y_C),len(self.Y_C)))*self.Y_C).T - (np.ones((len(self.Y_C),len(self.Y_C)))*self.Y_C))**2
        
        self.Dc_T = np.ones((self.Xc_T.shape[0],self.Xc_T.shape[1],self.Xc_T.shape[0])) * self.Xc_T.T
        self.Dc_T = (self.Dc_T - self.Dc_T.T) 
        self.Dc_C = np.ones((self.Xc_C.shape[0],self.Xc_C.shape[1],self.Xc_C.shape[0])) * self.Xc_C.T
        self.Dc_C = (self.Dc_C - self.Dc_C.T) 
        
        self.Dd_T = np.ones((self.Xd_T.shape[0],self.Xd_T.shape[1],self.Xd_T.shape[0])) * self.Xd_T.T
        self.Dd_T = (self.Dd_T != self.Dd_T.T) 
        self.Dd_C = np.ones((self.Xd_C.shape[0],self.Xd_C.shape[1],self.Xd_C.shape[0])) * self.Xd_C.T
        self.Dd_C = (self.Dd_C != self.Dd_C.T) 

    def threshold(self,x):
        k = self.k
        for i in range(x.shape[0]):
            row = x[i,:]
            row1 = np.where( row < row[np.argpartition(row,k+1)[k+1]],1,0)
            x[i,:] = row1
        return x
    
    def distance(self,Mc,Md,xc1,xd1,xc2,xd2):
        dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
        dd = np.sum((Md**2)*xd1!=xd2)
        return dc+dd
        
    def loss_(self, Mc, Md, xc1, xd1, y1, xc2, xd2, y2, gamma=1 ):
        w12 = np.exp( -1 * gamma * self.distance(Mc,Md,xc1,xd1,xc2,xd2) )
        return w12*((y1-y2)**2)
    
    def calcW_T(self,Mc,Md):
        #this step is slow
        Dc = np.sum( ( self.Dc_T * (Mc.reshape(-1,1)) )**2, axis=1)
        Dd = np.sum( ( self.Dd_T * (Md.reshape(-1,1)) )**2, axis=1)
        W = self.threshold( (Dc + Dd) )
        W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
        return W
    
    def calcW_C(self,Mc,Md):
        #this step is slow
        Dc = np.sum( ( self.Dc_C * (Mc.reshape(-1,1)) )**2, axis=1)
        Dd = np.sum( ( self.Dd_C * (Md.reshape(-1,1)) )**2, axis=1)
        W = self.threshold( (Dc + Dd) )
        W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
        return W
    
    def Delta_(self,Mc,Md):
        self.W_T = self.calcW_T(Mc,Md)
        self.W_C = self.calcW_C(Mc,Md)
        self.delta_T = np.sum((self.Y_T - (np.matmul(self.W_T,self.Y_T) - np.diag(self.W_T)*self.Y_T))**2)
        self.delta_C = np.sum((self.Y_C - (np.matmul(self.W_C,self.Y_C) - np.diag(self.W_C)*self.Y_C))**2)
        return self.delta_T + self.delta_C
    
    def objective(self,M):
        Mc = M[:len(self.continuous)]
        Md = M[len(self.continuous):]
        delta = self.Delta_(Mc,Md)
        reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 )
        cons1 = 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2
        cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md)) < 0 ) )
        return delta + reg + cons1 + cons2
        
    def fit(self,method='BFGS'):
        M_init = np.ones((self.p,))
        res = opt.minimize( self.objective, x0=M_init,method=method )
        self.M = res.x
        self.Mc = self.M[:len(self.continuous)]
        self.Md = self.M[len(self.continuous):]
        return res
    
    def get_matched_groups(self, df_estimation, k=10 ):
        #units to be matched
        Xc = df_estimation[self.continuous].to_numpy()
        Xd = df_estimation[self.discrete].to_numpy()
        Y = df_estimation[self.outcome].to_numpy()
        T = df_estimation[self.treatment].to_numpy()
        #splitted estimation data for matching
        df_T = df_estimation.loc[df_estimation[self.treatment]==1]
        df_C = df_estimation.loc[df_estimation[self.treatment]==0]
        #converting to numpy array
        Xc_T = df_T[self.continuous].to_numpy()
        Xc_C = df_C[self.continuous].to_numpy()
        Xd_T = df_T[self.discrete].to_numpy()
        Xd_C = df_C[self.discrete].to_numpy()
        Y_T = df_T[self.outcome].to_numpy()
        Y_C = df_C[self.outcome].to_numpy()
        D_T = np.zeros((Y.shape[0],Y_T.shape[0]))
        D_C = np.zeros((Y.shape[0],Y_C.shape[0]))
        #distance_treated
        Dc_T = (np.ones((Xc_T.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_T.shape[0])) * Xc_T.T).T)
        Dc_T = np.sum( (Dc_T * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
        Dd_T = (np.ones((Xd_T.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_T.shape[0])) * Xd_T.T).T )
        Dd_T = np.sum( (Dd_T * (self.Md.reshape(-1,1)) )**2 , axis=1 )
        D_T = (Dc_T + Dd_T).T
        #distance_control
        Dc_C = (np.ones((Xc_C.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_C.shape[0])) * Xc_C.T).T)
        Dc_C = np.sum( (Dc_C * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
        Dd_C = (np.ones((Xd_C.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_C.shape[0])) * Xd_C.T).T )
        Dd_C = np.sum( (Dd_C * (self.Md.reshape(-1,1)) )**2 , axis=1 )
        D_C = (Dc_C + Dd_C).T
        MG = {}
        for i in range(Y.shape[0]):
            #finding k closest control units to unit i
            idx = np.argpartition(D_C[i,:],k)
            matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C = Xc_C[idx[:k],:], Xd_C[idx[:k],:], Y_C[idx[:k]], D_C[i,idx[:k]]
            #finding k closest treated units to unit i
            idx = np.argpartition(D_T[i,:],k)
            matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T = Xc_T[idx[:k],:], Xd_T[idx[:k],:], Y_T[idx[:k]],D_T[i,idx[:k]]
            MG[i] = {'unit':[ Xc[i], Xd[i], Y[i], T[i] ] ,'control':[ matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C],'treated':[matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T ]}
        return MG
    
    def CATE(self,MG,outcome_discrete=False,model='linear'):
        cate = {}
        for k,v in MG.items():
            #control
            matched_X_C = np.hstack((v['control'][0],v['control'][1]))
            matched_Y_C = v['control'][2]
            #treated
            matched_X_T = np.hstack((v['treated'][0], v['treated'][1]))
            matched_Y_T = v['treated'][2]
            x = np.hstack(([v['unit'][0]], [v['unit'][1]]))
            if not outcome_discrete:
                if model=='mean':
                    yt = np.mean(matched_Y_T)
                    yc = np.mean(matched_Y_C)
                    cate[k] = {'CATE': yt - yc,'outcome':v['unit'][2],'treatment':v['unit'][3] }
                if model=='linear':
                    yc = lm.Ridge().fit( X = matched_X_C, y = matched_Y_C )
                    yt = lm.Ridge().fit( X = matched_X_T, y = matched_Y_T )
                    cate[k] = {'CATE': yt.predict(x) - yc.predict(x),'outcome':v['unit'][2],'treatment':v['unit'][3] }
                if model=='RF':
                    yc = ensemble.RandomForestRegressor().fit( X = matched_X_C, y = matched_Y_C )
                    yt = ensemble.RandomForestRegressor().fit( X = matched_X_T, y = matched_Y_T )
                    cate[k] = {'CATE': yt.predict(x)[0] - yc.predict(x)[0],'outcome':v['unit'][2],'treatment':v['unit'][3] }
        return pd.DataFrame.from_dict(cate,orient='index')
    
    def visualizeMG(self,MG,a):
        MGi = MG[a]
        k = len(MGi['control'][2])
        Xc = np.vstack( (MGi['control'][0],MGi['treated'][0]) )
        Xd = np.vstack( (MGi['control'][1],MGi['treated'][1]) )
        df = pd.DataFrame(np.hstack( (Xc,Xd) ))
        T = np.array([0 for i in range(0,k)] + [1 for i in range(0,k)])
        df['T'] = T
        df['Y'] = np.hstack( (MGi['control'][2],MGi['treated'][2]) )
        fig,axs = plt.subplots(nrows=int(np.ceil(len(df.columns)/4)),ncols=4,squeeze=False, sharey=True, figsize=(5*int(np.ceil(len(df.columns)/4)),20))
        for i, col in enumerate(df.columns):
            sns.scatterplot(x=col,y='Y',data=df,hue='T',ax=axs[int(i/4),i%4])
        plt.tight_layout()
        fig.savefig('matched_group_%d.png'%(a))
        fig = plt.Figure(figsize=(15,20))
        pd.plotting.parallel_coordinates(df,'T')
        fig.savefig('parallel_coordinate_matched_group_%d.png'%(a))
        return df
    
    def visualizeDimension(self,MG,x1,x2):
        X = []
        fig,ax = plt.subplots(1)
        for k,MGi in MG.items():
            x = np.hstack( (MGi['unit'][0],MGi['unit'][0]) )[ [ x1, x2 ] ]
            X.append(x)
            Xc = np.vstack( (MGi['control'][0],MGi['treated'][0]) )
            Xd = np.vstack( (MGi['control'][1],MGi['treated'][1]) )
            Xm = np.hstack((Xc,Xd))[:,[x1,x2]]
            x1min,x1max = np.min(Xm[:,0]), np.max(Xm[:,0])
            x2min,x2max = np.min(Xm[:,1]), np.max(Xm[:,1])
            rect = patches.Rectangle(x-np.array([(x1max-x1min)/2,(x2max-x2min)/2]),width=x1max-x1min,height=x2max-x2min,linewidth=1,edgecolor='r',facecolor='grey',alpha=0.06)
            ax.add_patch(rect)
        X = np.array(X)
        plt.scatter(X[:,0],X[:,1])
        fig.savefig('marginal_%d_%d_matched_groups.png')
        return X
        
            
        
        
        
        
        