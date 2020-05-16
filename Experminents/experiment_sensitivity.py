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

n = 200
p = 2
ays = np.arange(-2, 2, 0.25)
azs = np.arange(-2, 2, 0.25)

s = np.zeros((len(ays),len(azs)))

fig,ax = plt.subplots(len(ays),len(azs))
seeds = [i for i in range(0,5)]

for seed in seeds:
    for i in range(len(ays)):
        for j in range(len(azs)):
            ay = ays[i]
            az = azs[j]
            np.random.seed(seed)
            df, df_true = dg.sensitivity_datagen(n, p, ay, az)
            
            sns.scatterplot('X0','X1',hue='T',alpha=0.4,data=df,ax=ax[i,j],legend=False)
            ax[i,j].title.set_text('%.1f, %.1f'%(ay,az))
            
            m = pymalts.malts_mf('Y', 'T', df,k_tr=5,k_est=5,estimator='linear',smooth_cate=True,reweight=False,n_splits=2,n_repeats=1)
            cate = m.CATE_df['avg.CATE']
            ate = cate.mean()
            s[i,j] += (ate-1)
s = s/5

fig = plt.figure()
CS = plt.contourf(ays,azs,s)
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar()
plt.xlabel('sensitivity parameter for Outcome')
plt.ylabel('sensitivity parameter for Treatment')
plt.tight_layout()
fig.savefig('sensitivity_analysis.png')