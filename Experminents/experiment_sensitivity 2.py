# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:22:11 2020

@author: Harsh
"""


import numpy as np
import pandas as pd
import pymalts
import datagen as dg
import matplotlib.pyplot as plt
import seaborn as sns

n = 500
p = 2
ays = np.arange(-2, 3, 1)
azs = np.arange(-2, 3, 1)
seeds = [i for i in range(0,5)]

s = np.zeros((len(ays),len(azs),len(seeds)))

fig,ax = plt.subplots(len(ays),len(azs))

for seed in seeds:
    for i in range(len(ays)):
        for j in range(len(azs)):
            ay = ays[i]
            az = azs[j]
            np.random.seed(seed)
            df, df_true = dg.sensitivity_datagen(n, p, ay, az)
            
            sns.scatterplot('X0','X1',hue='T',alpha=0.4,data=df,ax=ax[i,j],legend=False)
            ax[i,j].title.set_text('%.1f, %.1f'%(ay,az))
            
            m = pymalts.malts_mf('Y', 'T', df,k_tr=5,k_est=20,estimator='linear',smooth_cate=True,reweight=False,n_splits=2,n_repeats=1)
            cate = m.CATE_df['avg.CATE']
            ate = cate.mean()
            s[i,j,seed] += (ate-1)
            
s = np.mean(s,axis=2)
sns.set()
fig = plt.figure()
CS = plt.contour(ays,azs,s*10-0.05,levels=50)
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar()
plt.xlabel('sensitivity parameter for Outcome')
plt.ylabel('sensitivity parameter for Treatment')
plt.tight_layout()
plt.axvline(0,c='r')
plt.axhline(0,c='r')
fig.savefig('Figures/sensitivity_analysis.png')