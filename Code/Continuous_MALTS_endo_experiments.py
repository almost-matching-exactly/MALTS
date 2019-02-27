#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:49:29 2018

@author: harshparikh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:55:55 2018

@author: Harsh
"""
import numpy as np
import scipy.optimize as opt
from sklearn import cluster as cluster
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
import time
import pandas as pd
import data_generation as dg
import distancemetriclearning as dml
import FLAMEbit as FLAME
import rpy2
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpack
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.lib as rlib
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import InsaneLearner
import cfr_net_train
#import ACIC_2
import warnings
warnings.filterwarnings('ignore')

K=10
currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('Logs/log'+currenttime+'.txt','w')

numExperiment = 1
n_array = [500]
p_imp_array = [8]
#p_nonimp_array = np.linspace(2,50,10)
for i in range(0,len(p_imp_array)):
    for j in range(0,len(n_array)):
        currenttime = time.asctime(time.localtime())
        currenttime = currenttime.replace(':','_')
        num_control = int(n_array[j])//2
        num_treated = int(n_array[j])//2
        num_cov_dense = int(p_imp_array[i])
        num_covs_unimportant = 10
        numExample = num_control + num_treated
        num_covariates = num_cov_dense+num_covs_unimportant
        numVariable = num_covariates
        
        print('num experiment: '+str(numExperiment)+', num examples: '+str(numExample)+', num covariates: '+str(numVariable), file=log)
            
        #Data Generation
            
        
        ##non-constant treatment continuous
        data = dg.data_generation_dense_endo(numExample, num_cov_dense, num_covs_unimportant,rho=0.2)
        df, dense_bs, treatment_eff_coef = data
        X,Y,T = np.array(df[df.columns[0:num_covariates]]), np.array(df['outcome']), np.array(df['treated'])
        n,m = X.shape
        dftest,_,_ = dg.data_generation_dense_2(2500, 2500, num_cov_dense, num_covs_unimportant)
        Xtest,Ytest,Ttest = np.array(dftest[dftest.columns[0:num_covariates]]), np.array(dftest['outcome']), np.array(dftest['treated'])
        t_true = np.dot((Xtest[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(Xtest[:,:num_cov_dense]), axis=1)
        discrete = False
        
        
        currenttime = time.asctime(time.localtime())
        currenttime = currenttime.replace(':','_')
        
        ##InsaneLearning
        insane = InsaneLearner.InsaneLearner(100)
        insane.insaneLearning(X,Y,T)
        t_hat_insane = insane.CATE(Xtest)
        discrete = False
        t_true_insane = t_true
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        identity_line = np.linspace(min(min(t_true_insane), min(t_hat_insane)), max(max(t_true_insane), max(t_hat_insane)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_insane,t_hat_insane,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_insane_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        
        #------------------------------------------------------------------------------------
        #MALTS Non-Gradient Optimization
        
        discrete = True
        adknn = dml.distance_metric_learning(m,diag=True)
        Dc,Dt = adknn.split(X,Y,T)
        Xc,Yc,Tc = Dc
        Xt,Yt,Tt = Dt
        optresult = adknn.optimize(X,Y,T,numbatch=1)
        
        Lstar = adknn.L
        fLC = lambda y: Lstar
        
        dfC = pd.DataFrame(Lstar)
        dcate_cobyla, t_hat_cobyla = adknn.CATE(Xtest,X,Y,T,fLC=fLC,fLT=fLC)
        t_true_cobyla = t_true
        print("L matrix", file=log)
        print(dfC, file=log)
        
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        identity_line = np.linspace(min(min(t_true_cobyla), min(t_hat_cobyla)), max(max(t_true_cobyla), max(t_hat_cobyla)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_cobyla,t_hat_cobyla,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_MALTS_NonGradient_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        
        def diameterMatchGroup(tup,L):
            xi = tup[2]
            Xs = np.vstack((tup[0][0],tup[0][1]))
            l_dis = [ adknn.distance(xi,xj,Lstar) for xj in Xs]
            return max(l_dis)
        err = []
        diam = []
        for i in range(0,len(t_hat_cobyla)):
            diam.append(1/diameterMatchGroup(dcate_cobyla[i],Lstar))
            err.append(np.abs(t_hat_cobyla[i]-t_true_cobyla[i]))

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        diam = np.array(diam)
        diam2 = diam.copy()
        diam2.sort()
        err = np.array(err)
        popt, pcov = curve_fit(func, diam, err)

        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        scatter(diam,err,alpha=0.1)
        plot(diam2, func(diam2, popt[0],popt[1],popt[2]), color="black", linestyle="dashed", linewidth=3.0)
        xlabel('1/Diameter of Matched group')
        ylabel('Absolute CATE Error')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_MALTS_Err_Diam_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        scatter(diam,err,alpha=0.1)
        plot(diam2, func(diam2, popt[0],popt[1],popt[2]), color="black", linestyle="dashed", linewidth=3.0)
        xlabel('1/Diameter of Matched group')
        ylabel('Absolute CATE Error')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_MALTS_Err_Diam_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        
        #-------------------------------------------------------------------------------------------
#        ##MALTS Gradient Optimization
#        discrete = True
#        adknnGD = dml.distance_metric_learning(m,discrete=discrete)
#        Dc,Dt = adknnGD.split(X,Y,T)
#        Xc,Yc,Tc = Dc
#        Xt,Yt,Tt = Dt
#        adknnGD.L = Lstar
#        optresult = adknnGD.optimizeGD(X,Y,T,numbatch=5,searchspace=np.diag(adknnGD.L).argsort()[-(num_covariates//2):][::-1])
#        
#        LstarGD = adknnGD.L
#        fLCGD = lambda y: LstarGD
#        
#        #Lg = np.zeros((num_covariates,num_covariates))
#        #for i in range(0,10):
#        #    Lg[i,i] = 1.0
#        #print( "Good L" ) 
#        #print( adknn.objective(Xc,Yc,Tc,Lg) + adknn.objective(Xt,Yt,Tt,Lg) )
#        #Lid = np.zeros((num_covariates,num_covariates))
#        #print( "Identity L" ) 
#        #for i in range(0,20):
#        #    Lid[i,i] = 1.0
#        #print( adknn.objective(Xc,Yc,Tc,Lid) + adknn.objective(Xt,Yt,Tt,Lid) )
#        #s = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#        #loss = []
#        #for m in s:
#        #    Lm = m*Lstar + (1-m)*Lid
#        #    loss.append( adknn.objective(Xc,Yc,Tc,Lm) + adknn.objective(Xt,Yt,Tt,Lm) )
#        ##    
#        #fig = figure(figsize=(10,10))
#        #
#        #plot(s, loss, color="black", linestyle="dashed", linewidth=2.0)
#        ##ylim((250000, 400000)) 
#        #fig.savefig('Figures/Loss_True_Mat.jpg')
#        
#        dfCGD = pd.DataFrame(LstarGD)
#        dcate_GD, t_hat_GD = adknnGD.CATE(Xtest,X,Y,T,fLC=fLCGD,fLT=fLCGD)
#        t_true_GD = t_true
#        print("L matrix", file=log)
#        print(dfCGD, file=log)
#        
#        
#        fig = figure(figsize=(10,10))
#        identity_line = np.linspace(min(min(t_true_GD), min(t_hat_GD)), max(max(t_true_GD), max(t_hat_GD)))
#        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
#        scatter(t_true_GD,t_hat_GD,alpha=0.15)
#        xlabel('True Treatment')
#        ylabel('Predicted Treatment')
#        fig.savefig('Figures/CATE_distancemetriclearning_MALTS_Gradient_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
#        
#        fig = figure(figsize=(10,10))
#        lossa = optresult[1]
#        for lossb in lossa:
#            plot(lossb)
#        xlabel('Iterations')
#        ylabel('Error')
#        legend(['Batch'+str(i) for i in range(0,len(lossa))])
#        fig.savefig('Figures/CATE_distancemetriclearning_MALTS_Gradient_Learning_rate_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'_'+str(currenttime)+'.jpg')
#        
        #---------------------------------------------------------------------------------------------
        #Causal Random Forest
        utils = importr('utils')
#        packnames = ('MatchIt','grf')
#        utils.install_packages(StrVector(packnames))
        base = importr('base')
        
        grf = importr('grf')
        Ycrf = Y.reshape((len(Y),1))
        Tcrf = T.reshape((len(T),1))
        crf = grf.causal_forest(X,Ycrf,Tcrf)
        tauhat = grf.predict_causal_forest(crf,Xtest)
        tau_hat = tauhat[0]
        t_hat_crf = list(tau_hat)
        t_true_crf = t_true
        #
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        identity_line = np.linspace(min(min(t_true_crf), min(t_hat_crf)), max(max(t_true_crf), max(t_hat_crf)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_crf,t_hat_crf,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_crf_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        #---------------------------------------------------------------------------------------------
        ##GenMatch
        df.to_csv('non-constant-treatment-continuous.csv',index=False)
        #dftest = pd.DataFrame(Xtest)
        dftest.to_csv('test-non-constant-treatment-continuous.csv',index=False)
        
        string = """
        library('MatchIt')
        
        mydata <- read.csv('test-non-constant-treatment-continuous.csv')
        
        r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7,
                     method = 'genetic', data = mydata)
        
        mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
        
        hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
        """
        genmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
        t_hat_genmatch = np.array(genmatch.hh)
        x_hat_genmatch = np.array(genmatch.mtch).T[:,:10]
        t_true_genmatch = np.dot((x_hat_genmatch[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_genmatch[:,:5]), axis=1)
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        identity_line = np.linspace(min(min(t_true_genmatch), min(t_hat_genmatch)), max(max(t_true_genmatch), max(t_hat_genmatch)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_genmatch,t_hat_genmatch,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_genmatch_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        
        #---------------------------------------------------------------------------------------------
        ##Nearest Neighbor Matching
        string = """
        library('MatchIt')
        
        mydata <- read.csv('test-non-constant-treatment-continuous.csv')
        
        r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7,
                     method = 'nearest', data = mydata)
        
        mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
        
        hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
        """
        nearestmatch = SignatureTranslatedAnonymousPackage(string, "powerpack")
        t_hat_nearestmatch = np.array(nearestmatch.hh)
        x_hat_nearestmatch = np.array(nearestmatch.mtch).T[:,:10]
        t_true_nearestmatch = np.dot((x_hat_nearestmatch[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_nearestmatch[:,:5]), axis=1)
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size': 36})
        identity_line = np.linspace(min(min(t_true_nearestmatch), min(t_hat_nearestmatch)), max(max(t_true_nearestmatch), max(t_hat_nearestmatch)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_nearestmatch,t_hat_nearestmatch,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_nearestmatch_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        ##
        
        #---------------------------------------------------------------------------------------------
#        ##Coarsened Exact Matching
#        string = """
#        library('MatchIt')
#        
#        mydata <- read.csv('test-non-constant-treatment-continuous.csv')
#        
#        r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7,
#                     method = 'cem', data = mydata)
#        
#        mtch <- mydata[as.numeric(names(r$match.matrix[,])),]
#        
#        hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']
#        """
#        cem = SignatureTranslatedAnonymousPackage(string, "powerpack")
#        t_hat_cem = np.array(cem.hh)
#        x_hat_cem = np.array(cem.mtch).T[:,:10]
#        t_true_cem = np.dot((x_hat_cem[:,:len(treatment_eff_coef)]),treatment_eff_coef) + np.sum(dg.construct_sec_order(x_hat_cem[:,:5]), axis=1)
#        
#        fig = figure(figsize=(10,10))
#        rcParams.update({'font.size': 36})
#        identity_line = np.linspace(min(min(t_true_cem), min(t_hat_cem)), max(max(t_true_cem), max(t_hat_cem)))
#        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
#        scatter(t_true_cem,t_hat_cem,alpha=0.15)
#        xlabel('True Treatment')
#        ylabel('Predicted Treatment')
#        tight_layout()
#        fig.savefig('Figures/CATE_distancemetriclearning_CEM_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
#        ##
#        
        #----------------------------------------------------------------------------------------------
        ##DBARTS
        dbarts = importr('dbarts')
        bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
        y_c_hat_bart = np.array(bart_res_c.rx(8))
        bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
        y_t_hat_bart = np.array(bart_res_t.rx(8))
        t_hat_bart = list(y_t_hat_bart - y_c_hat_bart)[0]
        t_true_bart = t_true
        
        fig = figure(figsize=(10,10))
        rcParams.update({'font.size':36})
        identity_line = np.linspace(min(min(t_true_bart), min(t_hat_bart)), max(max(t_true_bart), max(t_hat_bart)))
        plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)
        scatter(t_true_bart,t_hat_bart,alpha=0.15)
        xlabel('True Treatment')
        ylabel('Predicted Treatment')
        tight_layout()
        fig.savefig('Figures/CATE_distancemetriclearning_endo_BART_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')
        ##
        
        #----------------------------------------------------------------------------------------------
        
        delt_insane = np.array(list(map(np.abs,np.array(t_hat_insane) - np.array(t_true_insane))))
        delt_cobyla = np.array(list(map(np.abs,np.array(t_hat_cobyla) - np.array(t_true_cobyla))))
        delt_crf = np.array(list(map(np.abs,np.array(t_hat_crf) - np.array(t_true_crf))))
        delt_genmatch = np.array(list(map(np.abs,np.array(t_hat_genmatch) - np.array(t_true_genmatch))))
        delt_nearestmatch = np.array(list(map(np.abs,np.array(t_hat_nearestmatch) - np.array(t_true_nearestmatch))))
        #delt_cem = np.array(list(map(np.abs,np.array(t_hat_cem) - np.array(t_true_cem))))
        delt_bart = np.array(list(map(np.abs,np.array(t_hat_bart) - np.array(t_true_bart))))
        
        h_insane = np.histogram(delt_insane,bins=25)
        h_cobyla = np.histogram(delt_cobyla,bins=25)
        h_crf = np.histogram(delt_crf,bins=25)
        h_genmatch = np.histogram(delt_genmatch,bins=25)
        h_nearestmatch = np.histogram(delt_nearestmatch,bins=25)
        #h_cem = np.histogram(delt_cem,bins=25)
        h_bart = np.histogram(delt_bart,bins=25)
        
        fig = figure(figsize=(8.75,7))
        rcParams.update({'font.size': 22})
        rcParams['lines.linewidth'] = 5
        plot(h_insane[1][:-1],h_insane[0])
        fill_between(h_insane[1][:-1],h_insane[0],alpha=0.2)
        plot(h_cobyla[1][:-1],h_cobyla[0])
        fill_between(h_cobyla[1][:-1],h_cobyla[0],alpha=0.2)
        plot(h_crf[1][:-1],h_crf[0])
        fill_between(h_crf[1][:-1],h_crf[0],alpha=0.2)
#        plot(h_genmatch[1][:-1],h_genmatch[0])
#        fill_between(h_genmatch[1][:-1],h_genmatch[0],alpha=0.2)
#        plot(h_nearestmatch[1][:-1],h_nearestmatch[0])
#        fill_between(h_nearestmatch[1][:-1],h_nearestmatch[0],alpha=0.2)
#        plot(h_cem[1][:-1],h_cem[0])
#        fill_between(h_cem[1][:-1],h_cem[0],alpha=0.2)
        plot(h_bart[1][:-1],h_bart[0])
        fill_between(h_bart[1][:-1],h_bart[0],alpha=0.2)
        legend(['Diff. of RF','MALTS','Causal Forest','BART'])
        xlabel('CATE absolute error')
        ylabel('Frequency')
        tight_layout() 
        fig.savefig('Figures/error_cates_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')

        fig = figure(figsize=(8.75,6.6))
        rcParams.update({'font.size': 18})
        rcParams['lines.linewidth'] = 4
#        ylim(top=200)
        violinplot(delt_cobyla,positions=[1])
        violinplot(delt_genmatch,positions=[2])
        violinplot(delt_nearestmatch,positions=[3])
#        violinplot(delt_cem,positions=[4])
        violinplot(delt_bart,positions=[5])
        violinplot(delt_crf,positions=[6])
        violinplot(delt_insane,positions=[7])
        xticks([1,2,3,5,6,7],['MALTS','Genmatch','NN Matching','BART','Causal Forest','Diff. of RF'],rotation=75)
        ylabel('CATE absolute error')
        tight_layout()        
        fig.savefig('Figures/violin_error_cates_endo_'+str(num_control)+'_'+str(num_treated)+'_'+str(num_cov_dense)+'_'+str(num_covs_unimportant)+'.jpg')

        
log.close()
