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
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
warnings.filterwarnings("ignore")

class malts:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,k=10,reweight=False):
        # np.random.seed(0)
        self.C = C #coefficient to regularozation term
        self.k = k
        self.reweight = reweight
        self.n, self.p = data.shape
        self.p = self.p - 2 #shape of the data
        self.outcome = outcome
        self.treatment = treatment
        self.discrete = discrete
        self.continuous = list(set(data.columns).difference(set([outcome]+[treatment]+discrete)))
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
        if self.reweight == False:
            return self.delta_T + self.delta_C
        elif self.reweight == True:
            return (len(self.Y_T)+len(self.Y_C))*(self.delta_T/len(self.Y_T) + self.delta_C/len(self.Y_C))
    
    def objective(self,M):
        Mc = M[:len(self.continuous)]
        Md = M[len(self.continuous):]
        delta = self.Delta_(Mc,Md)
        reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 )
        cons1 = 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2
        cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md)) < 0 ) )
        return delta + reg + cons1 + cons2
        
    def fit(self,method='COBYLA'):
        # np.random.seed(0)
        M_init = np.ones((self.p,))
        res = opt.minimize( self.objective, x0=M_init,method=method )
        self.M = res.x
        self.Mc = self.M[:len(self.continuous)]
        self.Md = self.M[len(self.continuous):]
        self.M_opt = pd.DataFrame(self.M.reshape(1,-1),columns=self.continuous+self.discrete,index=['Diag'])
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
        index = df_estimation.index
        for i in range(Y.shape[0]):
            #finding k closest control units to unit i
            idx = np.argpartition(D_C[i,:],k)
            matched_df_C = pd.DataFrame( np.hstack( (Xc_C[idx[:k],:], Xd_C[idx[:k],:].reshape((k,len(self.discrete))), Y_C[idx[:k]].reshape(-1,1), D_C[i,idx[:k]].reshape(-1,1), np.zeros((k,1)) ) ), index=df_C.index[idx[:k]], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] )
            #finding k closest treated units to unit i
            idx = np.argpartition(D_T[i,:],k)
            matched_df_T = pd.DataFrame( np.hstack( (Xc_T[idx[:k],:], Xd_T[idx[:k],:].reshape((k,len(self.discrete))), Y_T[idx[:k]].reshape(-1,1), D_T[i,idx[:k]].reshape(-1,1), np.ones((k,1)) ) ), index=df_T.index[idx[:k]], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] )
            matched_df = pd.DataFrame(np.hstack((Xc[i], Xd[i], Y[i], 0, T[i])).reshape(1,-1), index=['query'], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment])
            matched_df = matched_df.append(matched_df_T.append(matched_df_C))
            MG[index[i]] = matched_df
            #{'unit':[ Xc[i], Xd[i], Y[i], T[i] ] ,'control':[ matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C],'treated':[matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T ]}
        MG_df = pd.concat(MG)
        return MG_df
    
    def CATE(self,MG,outcome_discrete=False,model='linear'):
        cate = {}
        for k in pd.unique(MG.index.get_level_values(0)):
            v = MG.loc[k]
            #control
            matched_X_C = v.loc[v[self.treatment]==0].drop(index='query',errors='ignore')[self.continuous+self.discrete]
            matched_Y_C = v.loc[v[self.treatment]==0].drop(index='query',errors='ignore')[self.outcome]
            #treated
            matched_X_T = v.loc[v[self.treatment]==1].drop(index='query',errors='ignore')[self.continuous+self.discrete]
            matched_Y_T = v.loc[v[self.treatment]==1].drop(index='query',errors='ignore')[self.outcome]
            x = v.loc['query'][self.continuous+self.discrete].to_numpy().reshape(1,-1)
            
            vc = v[self.continuous].to_numpy()
            vd = v[self.discrete].to_numpy()
            dvc = np.ones((vc.shape[0],vc.shape[1],vc.shape[0])) * vc.T
            dist_cont = np.sum( ( (dvc - dvc.T) * (self.Mc.reshape(-1,1)) )**2, axis=1) 
            dvd = np.ones((vd.shape[0],vd.shape[1],vd.shape[0])) * vd.T
            dist_dis = np.sum( ( (dvd - dvd.T) * (self.Md.reshape(-1,1)) )**2, axis=1) 
            dist_mat = dist_cont + dist_dis
            diameter = np.max(dist_mat)
            
            if not outcome_discrete:
                if model=='mean':
                    yt = np.mean(matched_Y_T)
                    yc = np.mean(matched_Y_C)
                    cate[k] = {'CATE': yt - yc,'outcome':v.loc['query'][self.outcome],'treatment':v.loc['query'][self.treatment],'diameter':diameter }
                if model=='linear':
                    yc = lm.Ridge().fit( X = matched_X_C, y = matched_Y_C )
                    yt = lm.Ridge().fit( X = matched_X_T, y = matched_Y_T )
                    cate[k] = {'CATE': yt.predict(x)[0] - yc.predict(x)[0], 'outcome':v.loc['query'][self.outcome],'treatment':v.loc['query'][self.treatment],'diameter':diameter }
                if model=='RF':
                    yc = ensemble.RandomForestRegressor().fit( X = matched_X_C, y = matched_Y_C )
                    yt = ensemble.RandomForestRegressor().fit( X = matched_X_T, y = matched_Y_T )
                    cate[k] = {'CATE': yt.predict(x)[0] - yc.predict(x)[0], 'outcome':v.loc['query'][self.outcome],'treatment':v.loc['query'][self.treatment],'diameter':diameter }
        return pd.DataFrame.from_dict(cate,orient='index')
    
    def visualizeMG(self,MG,a):
        MGi = MG.loc[a]
        k = int( (MGi.shape[0] - 1 )/2 )
        df = MGi[self.continuous+self.discrete].drop(index='query')
        
        df.index.names = ['Unit']
        df.columns.names = ['Covariate']
        tidy = df.stack().to_frame().reset_index().rename(columns={0: 'Covariate Value'})  
        
        T = np.array([0 for i in range(0,k*self.p)] + [1 for i in range(0,k*self.p)])
        tidy[self.treatment] = T
        
        y0 = np.ones((self.p,k)) * MGi.loc[MGi[self.treatment]==0][self.outcome].drop(index='query',errors='ignore').to_numpy()
        y0 = y0.flatten('F')
        y1 = np.ones((self.p,k)) * MGi.loc[MGi[self.treatment]==1][self.outcome].drop(index='query',errors='ignore').to_numpy()
        y1 = y0.flatten('F')
        tidy[self.outcome] = np.hstack( (y0,y1) )
        fig = plt.figure()
        sns.lmplot(sharey=False,sharex=False,x='Covariate Value',y=self.outcome,hue=self.treatment, col='Covariate', data=tidy, col_wrap=3, height=4)
        fig.savefig('matched_group_%d.png'%(a))
        
        fig = plt.figure(figsize=(15,20))
        pd.plotting.parallel_coordinates(df,self.treatment,colormap=plt.cm.Set1)
        fig.savefig('parallel_coordinate_matched_group_%d.png'%(a))
        
        return tidy
    
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
            
        
class malts_mf:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,k_tr=15,k_est=50,estimator='linear',smooth_cate=True,reweight=False,n_splits=5,n_repeats=1,output_format='brief'):
        self.n_splits = n_splits
        self.C = C
        self.k_tr = k_tr
        self.k_est = k_est
        self.outcome = outcome
        self.treatment = treatment
        self.discrete = discrete
        self.continuous = list(set(data.columns).difference(set([outcome]+[treatment]+discrete)))
        self.reweight = reweight
        
        skf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
        gen_skf = skf.split(data,data[treatment])
        self.M_opt_list = []
        self.MG_list = []
        self.CATE_df = pd.DataFrame()
        N = np.zeros((data.shape[0],data.shape[0]))
        self.MG_matrix = pd.DataFrame(N, columns=data.index, index=data.index)
        
        i = 0
        for est_idx, train_idx in gen_skf:
            df_train = data.iloc[train_idx]
            df_est = data.iloc[est_idx]
            m = malts( outcome, treatment, data=df_train, discrete=discrete, C=self.C, k=self.k_tr, reweight=self.reweight )
            m.fit()
            self.M_opt_list.append(m.M_opt)
            mg = m.get_matched_groups(df_est,k_est)
            self.MG_list.append(mg)
            self.CATE_df = pd.concat([self.CATE_df, m.CATE(mg,model=estimator)], join='outer', axis=1)
        
        for i in range(n_splits*n_repeats):
            mg_i = self.MG_list[i]
            for a in mg_i.index:
                if a[1]!='query':
                    self.MG_matrix.loc[a[0],a[1]] = self.MG_matrix.loc[a[0],a[1]]+1
        
        cate_df = self.CATE_df['CATE']
        cate_df['avg.CATE'] = cate_df.mean(axis=1)
        cate_df['std.CATE'] = cate_df.std(axis=1)
        cate_df[self.outcome] = self.CATE_df['outcome'].mean(axis=1)
        cate_df[self.treatment] = self.CATE_df['treatment'].mean(axis=1)
        cate_df['avg.Diameter'] = self.CATE_df['diameter'].mean(axis=1)
        
        LOWER_ALPHA = 0.05
        UPPER_ALPHA = 0.95
        # Each model has to be separate
        lower_model = ensemble.GradientBoostingRegressor(loss='quantile',alpha=LOWER_ALPHA)
        # The mid model will use the default loss
        mid_model = ensemble.GradientBoostingRegressor(loss="ls")
        upper_model = ensemble.GradientBoostingRegressor(loss="quantile",alpha=UPPER_ALPHA)
        
        try:
            lower_model.fit(data[self.continuous+self.discrete], cate_df['avg.CATE'])
            mid_model.fit(data[self.continuous+self.discrete], cate_df['avg.CATE'])
            upper_model.fit(data[self.continuous+self.discrete], cate_df['avg.CATE'])
            
            cate_df['std.gbr.CATE'] = np.abs(upper_model.predict(data[self.continuous+self.discrete]) - lower_model.predict(data[self.continuous+self.discrete]))/4
            cate_df['avg.gbr.CATE'] = mid_model.predict(data[self.continuous+self.discrete])
            
            # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, np.std(cate_df['avg.CATE'])))
            # gaussian_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100,normalize_y=True)
            # gaussian_model.fit(data[self.continuous+self.discrete], cate_df['avg.CATE'])
            # gp_pred = gaussian_model.predict(data[self.continuous+self.discrete], return_std=True)
            # cate_df['std.gp.CATE'] = gp_pred[1]
            # cate_df['avg.gp.CATE'] = gp_pred[0]
            
            # bayes_ridge = lm.BayesianRidge(fit_intercept=True)
            # bayes_ridge.fit(data[self.continuous+self.discrete], cate_df['avg.CATE'])
            # br_pred = bayes_ridge.predict(data[self.continuous+self.discrete], return_std=True)
            # cate_df['std.br.CATE'] = br_pred[1]
            # cate_df['avg.br.CATE'] = br_pred[0]
            
            if smooth_cate:
                cate_df['avg.CATE'] = cate_df['avg.gbr.CATE']
            cate_df['std.CATE'] = cate_df['std.gbr.CATE']
            
            if output_format=='brief':
                self.CATE_df = cate_df[['avg.CATE','std.CATE',outcome,treatment]]
            if output_format=='full':
                self.CATE_df = cate_df
        except:
            self.CATE_df = cate_df