# -*- coding: utf-8 -*-
"""
Created on Sat May 05 14:25:40 2018

@author: Harsh
"""

labelmap = {'S3':'Self-reported expectations for success in the future',
'C1':'Race or ethnicity',
'C2':'Gender',
'C3':'First-generation status',
'XC':'Urbanicity of the school',
'X1':'School-level mean of students fixed mindsets',
'X2':'School achievement level',
'X3':'School racial/ethnic minority composition',
'X4':'School poverty concentration',
'X5':'School size',
'T-hat': 'Predicted Individual Treatment Effects',
'Y': 'Y',
'diam':'Diameter'}


import numpy as np
import scipy.optimize as opt
from sklearn import cluster as cluster
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
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import FLAMEbit as FLAME
import warnings
warnings.simplefilter("ignore")
session = np.random.randint(0,10000)
currenttime = time.asctime(time.localtime())
currenttime = currenttime.replace(':','_')
log = open('Logs/ACIC_log'+currenttime+'.csv','w')

def read_file():
    f = 'Data/synthetic_data.csv'
    df = pd.read_csv(f)
    return df

def encode(df):
    df = pd.get_dummies(df,columns=['S3','C1','C2','C3','XC'])
    return df

def normalize(df):
    Z = df['Z']
    scid = df['schoolid']
    for col in df.columns:
        mean = np.mean(df[col])*np.ones(np.shape(df[col]))
        std = np.std(df[col])
        df[col] = (df[col] - mean)/std
    df['Z'] = Z
    df['schoolid'] = scid
    return df

def plot2dim(X,Y,labels=['X','Y'],kernel='rbf',degree=1,scatter=True,violin=False,box=False,xlim=None,ylim=None):
    d = {}
    n = len(Y)
    for i in range(0,n):
        k = (X[i])
        if k not in d:
            d[k] = []
        d[k] = d[k] +[Y[i]]
    X2,Y2 = [],[]
    for k,v in d.items():
        x = k
        y = np.mean(v)
        X2.append(x)
        Y2.append(y)
    X1,Y1 = X, Y
    svr = SVR(kernel=kernel, degree=degree,epsilon=0.775,C=0.65)
    X11 = np.reshape(np.array(X1),(-1, 1))
    svr = svr.fit(X11,Y1)
    xspace = np.linspace(min(X1), max(X1), 1000).reshape(-1, 1)
    yspace = svr.predict(xspace)
    fig = plt.figure(figsize=(8.75,6.6))
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['lines.linewidth'] = 4
    #plt.scatter(X2,Y2,c='black')
    if scatter:
        plt.scatter(X1,Y1,alpha=0.08)
    if violin:
        i = 1
        for k,v in d.items():
            plt.violinplot(v,positions=[k])
            i += 1
    if box:
        i = 1
        for k,v in d.items():
            plt.boxplot(v,positions=[k])
            i += 1
    if box or violin:
        #plt.scatter(X2,Y2,c='black')
        plt.xlim((0,min(d.keys())+len(d)+1))
        plt.xticks(list(d.keys()),list(d.keys()),rotation=75)
    if not ( violin or box ):
        plt.plot(xspace,yspace,c='black')
        #plt.ylim((min(yspace),max(yspace)))
    plt.xlabel(labelmap[labels[0]])
    plt.ylabel(labelmap[labels[1]])
    plt.tight_layout()
    fig.savefig('Figures/ACIC_'+labels[0]+'_'+labels[1]+'_'+str(session)+'.jpg')

    
def plotcontour(X,Y,Z,labels=['X','Y']):
    svr = SVR()
    X11 = np.vstack((X,Y)).T
    svr = svr.fit(X11,Z)
    x1space = np.mgrid[min(X):max(X):0.5, min(Y):max(Y):0.5]
    shp = np.shape(x1space)
    x11space = x1space.reshape(2,-1).T
    zspace = np.array(svr.predict(x11space))
    fig = plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['lines.linewidth'] = 4
    z = zspace.reshape((shp[2],shp[1]))
    cf = plt.contour(x1space[0,:,0],x1space[1,0,:],z)
    plt.xlabel(labelmap[labels[0]])
    plt.ylabel(labelmap[labels[1]])
    plt.clabel(cf)
    plt.tight_layout()
    fig.savefig('Figures/ACIC_'+labels[0]+'_'+labels[1]+'_'+str(session)+'.jpg')
    

def plot3dim(X,Y,Z,labels=['X','Y','Z']):
    d = {}
    n = len(Z)
    for i in range(0,n):
        k = (X[i],Y[i])
        if k not in d:
            d[k] = []
        d[k] = d[k] +[Z[i]]
    X1,Y1,Z1 = [],[],[]
    for k,v in d.items():
        x,y = k
        z = np.mean(v)
        X1.append(x)
        Y1.append(y)
        Z1.append(z)
    svr = SVR()
    X11 = np.vstack((X1,Y1)).T
    svr = svr.fit(X11,Z1)
    x11space = np.mgrid[min(X1):max(X1):0.5, min(Y1):max(Y1):0.5].reshape(2,-1).T
    zspace = svr.predict(x11space)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_trisurf(np.array(X1),np.array(Y1),np.array(Z1),alpha=0.4)
    ax.plot_trisurf(x11space[:,0],x11space[:,1],zspace,alpha=0.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.tight_layout()
    fig.savefig('Figures/ACIC_'+labels[0]+'_'+labels[1]+'_'+labels[2]+'_'+str(session)+'.jpg')
    
    
def unnormalize(df):
    df1 = read_file()
    df1 = encode(df1)
    Z = df['Z']
    scid = df['schoolid']
    for col in df.columns:
        mean = np.mean(df1[col])*np.ones(np.shape(df[col]))
        std = np.std(df1[col])
        df[col] = df[col]*std + mean
    df['Z'] = Z
    df['schoolid'] = scid
    return df

def split_discrete(X,d_columns):
    n,m = X.shape
    c_columns = set(np.arange(0,m)) - set(d_columns)
    X_discrete = np.array([X[:,i] for i in d_columns])
    X_continuous = np.array([X[:,i] for i in c_columns])
    X_discrete = X_discrete.T
    X_continuous = X_continuous.T
    return X_discrete, X_continuous

def club_discrete(X,d_columns):
    X_discrete, X_continuous = split_discrete(X,d_columns)
    d = {}
    for i in range(0,n):
        if str(X_discrete[i,:]) not in d:
            d[str(X_discrete[i,:])] = [],[],[]
        d[str(X_discrete[i,:])] = d[str(X_discrete[i,:])] + [X_continuous[i,:]]
    return d

def belongs(colid,s1,s2):
    s21 = [ s2[i] for i in colid ]
    return np.array_equal(s1,s21)
    
def diameterMatchGroup(tup,L):
    xi = tup[2]
    Xs = np.vstack((tup[0][0],tup[0][1]))
    l_dis = [ adknn.distance(xi,xj,Lstar) for xj in Xs]
    return max(l_dis)

df = read_file()
df_shuffle = shuffle(df)
df_shuffle_norm = df_shuffle
Xb,Yb,Tb = np.array(df_shuffle_norm[df_shuffle_norm.columns[3:]]), np.array(df_shuffle_norm['Y']), np.array(df_shuffle_norm['Z'])
n,m = Xb.shape

X,Y,T = Xb[:500,:],Yb[:500],Tb[:500]
d_columns = list(np.arange(0,5,dtype=int))
Xtest = Xb[500:,]

#MALTS
adknn = dml.distance_metric_learning(m,diag=True,dcov=d_columns,K=10,L=None,distance=dml.d)
optresult = adknn.optimize(X,Y,T,numbatch=1)
Lstar = adknn.L

fLC = lambda y: Lstar
dcate_cobyla, t_hat_cobyla = adknn.CATE(Xtest,Xb,Yb,Tb,fLC=fLC)
diam = []
diam2 = []
for i in range(0,len(t_hat_cobyla)):
    diam.append(np.exp(-0.1*diameterMatchGroup(dcate_cobyla[i],Lstar) ) )
    diam2.append(diameterMatchGroup(dcate_cobyla[i],Lstar))
fig = plt.figure()
plt.hist(diam2,bins=25)
plt.xlabel('Diameter of Matched Groups')
plt.ylabel('Frequency')
plt.tight_layout()
fig.savefig('Figures/ACIC_Hist_Diam.png')

def d_prune(Xb,t_hat_cobyla,diam,percentile=None,threshold=None):
    if threshold is None:
        if percentile is None:
            percentile = 100
        threshold = np.percentile(diam,q=percentile)      
    t_hat_prune = []
    Xb_prune = []
    for i in range(0,len(t_hat_cobyla)):
        if diam2[i]<threshold:
            t_hat_prune.append(t_hat_cobyla[i])
            Xb_prune.append(Xb[i,:])
    t_hat_prune = np.array(t_hat_prune)
    Xb_prune = np.array(Xb_prune)
    return t_hat_prune,Xb_prune

fig = plt.figure()
plt.scatter(diam2,t_hat_cobyla)
fig.savefig('Figures/ACIC_diam_cate_Scatter.png')
diam = np.mean(diam)
ATE_malts = np.mean(t_hat_cobyla)
ATE_malts_w = np.sum(np.array(diam)*np.array(t_hat_cobyla))/np.sum(diam)
#np.savetxt(log,t_hat_cobyla)

#BART
C,T = adknn.split(X,Y,T)
Xc,Yc,Tc, = C
Xt,Yt, Tt = T
dbarts = importr('dbarts')
bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
y_c_hat_bart = np.array(bart_res_c.rx(8))
bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
y_t_hat_bart = np.array(bart_res_t.rx(8))
t_hat_bart = list(y_t_hat_bart - y_c_hat_bart)[0]
ATE_bart = np.mean(t_hat_bart)

print('ATE MALTS,'+str(ATE_malts),file=log )
print('ATE BART,'+str(ATE_bart),file=log )
print('MALTS L-star',file=log)
np.savetxt(log,Lstar,delimiter=',')
print('T-hat (malts,bart)',file=log)
np.savetxt(log,np.array([t_hat_cobyla,t_hat_bart]),delimiter=',')

a = ['S3','C1','C2','C3','XC']
for i in range(len(a)):
    plot2dim(Xb_prune[:,i],t_hat_prune,labels=[a[i],'T-hat'],scatter=False,violin=True,box=True)
    
b = ['X1','X2','X3','X4','X5']
for i in range(len(b)):
    plot2dim(Xb_prune[:,5+i],t_hat_prune,labels=[b[i],'T-hat'],scatter=False,degree=2)
    
fig = plt.figure()
plt.hist(t_hat_prune)
fig.savefig('Figures/ACIC_hist_cates_prune.png')
log.close()

fig = plt.figure()
plt.scatter(Xb[:,4],Xb[:,5],c=t_hat_cobyla/max(t_hat_cobyla))
plt.legend()
fig.savefig('Figures/ACIC_X1_XC.png')

def matrixplot(mat,label=['X','Y'],remark=''):
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap=plt.cm.Blues)
    plt.xticks(np.arange(0,7),np.arange(1,8))
    plt.yticks(np.arange(0,5),np.arange(0,5))
    n,m = mat.shape
    for i in range(0,m):
        for j in range(0,n):
            c = mat[j,i]
            ax.text(i, j, str(round(c, 2)), va='center', ha='center')
    plt.xlabel(labelmap[label[0]])
    plt.ylabel(labelmap[label[1]])
    fig.savefig('Figures/ACIC_Mat_'+label[0]+'_'+label[1]+'_'+remark+'.png')
    
def multiplot2dim(Xa,Ya,labels=['X','Y'],kernel='rbf',degree=1,scatter=True,violin=False,box=False,numline=1,legend=None):
    fig = plt.figure(figsize=(8.75,6.6))
    for itr in range(0,numline):
        X, Y = Xa[itr], Ya[itr]
        d = {}
        n = len(Y)
        for i in range(0,n):
            k = (X[i])
            if k not in d:
                d[k] = []
            d[k] = d[k] +[Y[i]]
        X2,Y2 = [],[]
        for k,v in d.items():
            x = k
            y = np.mean(v)
            X2.append(x)
            Y2.append(y)
        X1,Y1 = X, Y
        svr = SVR(kernel=kernel, degree=degree,epsilon=0.775,C=0.65)
        X11 = np.reshape(np.array(X1),(-1, 1))
        svr = svr.fit(X11,Y1)
        xspace = np.linspace(min(X1), max(X1), 1000).reshape(-1, 1)
        yspace = svr.predict(xspace)
        
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['lines.linewidth'] = 4
        #plt.scatter(X2,Y2,c='black')
        if scatter:
            plt.scatter(X1,Y1,alpha=0.08)
        if violin:
            i = 1
            for k,v in d.items():
                plt.violinplot(v,positions=[k])
                i += 1
        if box:
            i = 1
            for k,v in d.items():
                plt.boxplot(v,positions=[k])
                i += 1
        if box or violin:
            #plt.scatter(X2,Y2,c='black')
            plt.xlim((min(d.keys())-1,min(d.keys())+len(d)+1))
            plt.xticks(list(d.keys()),list(d.keys()),rotation=75)
        if not ( violin or box ):
            plt.plot(xspace,yspace)
            #plt.ylim((min(yspace),max(yspace)))
    if legend is not None:
        plt.legend(legend)
    plt.xlabel(labelmap[labels[0]])
    plt.ylabel(labelmap[labels[1]])
    plt.tight_layout()
    fig.savefig('Figures/ACIC_multi_'+labels[0]+'_'+labels[1]+'_'+str(session)+'.jpg')

#err = []

#X_d, X_c = split_discrete(Xb,d_columns)
#n_d,m_d = X_d.shape
#df_d = pd.DataFrame(X_d)
#df_d['outcome'] = Yb
#df_d['treated'] = Tb
#df_d['matched'] = np.zeros((n_d,))
#
#resFlame = FLAME.run_bit(df_d, df_d, list(range(m_d)), [2]*m_d, tradeoff_param = 1)
#resdf = resFlame[1]
#d_set_tuple = []
#for temp_df in resdf:
#    for index, row in temp_df.iterrows():
#        cons_tup = ( list(temp_df.columns)[:-2], np.array(row)[:-2], [], [], [], dml.distance_metric_learning(m-m_d,discrete=False))
#        d_set_tuple.append(cons_tup)
#
#for i in range(0,n):
#    for j in range(0,len(d_set_tuple)):
#        tup = d_set_tuple[j]
#        if belongs(tup[0],tup[1],X_d[i,:]):
#            d_set_tuple[j] = (tup[0],tup[1],tup[2]+[X_c[i,:]],tup[3]+[Yb[i]],tup[4]+[Tb[i]],tup[5])
#            
#for j in range(0,len(d_set_tuple)):
#    print("Tuple-"+str(j))
#    tup = d_set_tuple[j]
#    tupres = tup[5].optimize(np.array(tup[2]),np.array(tup[3]),np.array(tup[4]),numbatch = 1)
#    d_set_tuple[j] = (tup[0],tup[1],tup[2],tup[3],tup[4],tup[5])
#            
#
##CTE = 0
##cnt = 0
##for j in range(0,len(d_set_tuple)):
##    print("Tuple-"+str(j))
##    tup = d_set_tuple[j]
##    LstarC_cobyla = tup[5].L
##    fLCcobyla = lambda y: LstarC_cobyla
##    dcate_cobyla, t_hat_cobyla = tup[5].CATE(np.array(tup[2]),np.array(tup[2]),np.array(tup[3]),np.array(tup[4]),fLC=fLCcobyla)
##    d_set_tuple[j] = (tup[0],tup[1],tup[2],tup[3],tup[4],tup[5],dcate_cobyla, t_hat_cobyla)
##    CTE += len(t_hat_cobyla)*np.average(t_hat_cobyla)
##    cnt += len(t_hat_cobyla)
##
##ATE = CTE/cnt
#
#ntest,mtest = Xtest.shape
#X_test_d, X_test_c = split_discrete(Xtest,d_columns)
#t_hat_cobyla = []
#w_hat_cobyla = []
#dcate_cobyla = {}
#for i in range(0,n):
#    t_hat_i = []
#    w_hat_i = []
#    dcate_i = {}
#    for j in range(0,len(d_set_tuple)):
#        tup = d_set_tuple[j]
#        if belongs(tup[0],tup[1],X_test_d[i,:]):
#            Lstar = tup[5].L
#            fL = lambda y: Lstar
#            dcate_ij, t_hat_ij = tup[5].CATE(np.array([X_test_c[i,:]]),np.array(tup[2]),np.array(tup[3]),np.array(tup[4]),fL)
#            w_ij = 1
#            #w_ij = 1/(diameterMatchGroup(dcate_ij[0],Lstar)**0)
#            t_hat_i += [t_hat_ij[0]*w_ij]
#            w_hat_i += [w_ij]
#            dcate_i[j] = dcate_ij[0]
#    val = np.average(t_hat_i)/np.average(w_hat_i) 
#    if not np.isnan(val):
#        t_hat_cobyla.append( np.average(t_hat_i)/np.average(w_hat_i) )
#        w_hat_cobyla.append(np.min(w_hat_i))
#        dcate_cobyla[i] = dcate_i
#ATE_cobyla = np.average(t_hat_cobyla)
#    
#
##adknn_cobyla = dml.distance_metric_learning(m,discrete=False)
##optresult = adknn_cobyla.optimize(X,Y,T,numBatch=10)
###optresult = adknn_cobyla.optimize_parallel(X,Y,T,iterations=5,numbatch=10) #int(np.sqrt(numExample))
##LstarC_cobyla = adknn_cobyla.Lc
##LstarT_cobyla = adknn_cobyla.Lt
##fLCcobyla = lambda y: LstarC_cobyla
##fLTcobyla = lambda y: LstarT_cobyla
###dcobyla = adknn_cobyla.nearestneighbormatching(X,Y,T,fLcobyla)
###ATEcobyla = adknn_cobyla.ATE(dcobyla)
##dfC_cobyla = pd.DataFrame(LstarC_cobyla)
##dfT_cobyla = pd.DataFrame(LstarT_cobyla)
##dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(Xtest,Xb,Yb,Tb,fLC=fLCcobyla,fLT=fLTcobyla)
###dcate_cobyla, t_hat_cobyla = adknn_cobyla.CATE(X,X,Y,T,fLcobyla)
##
##print("L matrix Control", file=log)
##print(dfC_cobyla, file=log)
##print("L matrix Treated", file=log)
##print(dfT_cobyla, file=log)
##Lstar = (LstarC_cobyla + LstarT_cobyla)/2
##print("L matrix averaged", file=log)
##print(pd.DataFrame(Lstar), file=log)
###print>>log, 'ATE COBYLA:- '+str(i)+' : '+str( ATEcobyla)
###ATEcobylaArray.append(ATEcobyla)
##ATE = np.mean(t_hat_cobyla)
##df_shuffle = unnormalize(df_shuffle_norm)
##
##Xb,Yb,Tb = np.array(df_shuffle[df_shuffle.columns[3:]]), np.array(df_shuffle['Y']), np.array(df_shuffle['Z'])
##plot3dim(Xb[:,0],Xb[:,1],t_hat_cobyla,labels=['X1','X2','T_hat'])
##plot2dim(Xb[:,0],t_hat_cobyla,labels=['X1','T_hat'])
##plot2dim(Xb[:,1],t_hat_cobyla,labels=['X2','T_hat'],degree=2)
##plotcontour(Xb[:,0],Xb[:,1],t_hat_cobyla,labels=['X1','X2'])
##
##Lstar_df = pd.DataFrame(Lstar)
##Lstar_df.to_csv('ACIC Results/L_.csv')
#
#def prettyprint(d):
#    f = open('ACIC matched groups PSD.csv','w')
#    print( *(['Unit'] + ['Treated'] + list(df_shuffle_norm.columns[3:]) + ['Y']), sep=', ', file = f )
#    for k,v in d.items():
#        xs, ys, ms = v
#        xsc, xst = xs
#        ysc, yst = ys
#        print( *([k] + [-1] + list(ms) + [0]), sep=', ', file = f )
#        for i in range(0,10):
#            print( *([k] + [0] + list(xsc[i]) + [ysc[i]]), sep=', ', file = f )
#        for i in range(0,10):
#            print( *([k] + [1] + list(xst[i]) + [yst[i]]), sep=', ', file = f )
#    f.close()
#        
#            