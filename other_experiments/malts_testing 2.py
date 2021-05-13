# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:28:31 2020

@author: ag520
"""
#%% Imports
import pandas as pd
import pymalts2 as pymalts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("C:/Users/angik/ANGIKAR/ANGIKAR_AME/CynthiaRudin_AME_Project/Additional_Functions/example.csv")
#%%
np.random.seed(0)
m = pymalts.malts_mf( outcome='outcome', treatment='treated', data=df)
ordint=sorted(m.MG_matrix.index.values)

#%%
def ATT(pym):
    return (np.mean(pym.CATE_df.loc[pym.CATE_df['treated']==1 ]['avg.CATE']))

print (ATT(m))

#%%This is same as ATE defined as vol0 for FLAME, and given in MALTS documentation.
def ATE_v0(pym):
    return pym.CATE_df['avg.CATE'].mean()

print (ATE_v0(m))
#%%
def mmgdict(pym):
    mgdict=dict()
    for i in ordint:
        max_series=pym.MG_matrix.loc[i]
        max_val=max(max_series)
        max_series=max_series[max_series==max_val]
        max_entries=sorted(max_series.index.values)
        max_group=df.loc[max_entries]
        mgdict[i]=max_group
    return mgdict

MMG_dict=mmgdict(m)

#%%
def mmgalldictweights(pym):
    mmgalldict=dict()
    for i in ordint:
        serie=pym.MG_matrix.loc[i]
        serie=serie[serie>0]
        entrie=serie.index.values
        mmgalldict[i]=(entrie, serie)
    return mmgalldict

MMG_dictall=mmgalldictweights(m)

#%%PROBLEM
def fakeCATE2(pym):
    CATEcate=[]
    for i in ordint:
        wt=MMG_dictall[i][1]
        tr=[]
        cn=[]
        trw=[]
        cnw=[]
        for j in wt.index.values:
            if df.loc[j, 'treated']==1:
                tr.append((df.loc[j, 'outcome'])*wt[j])
                trw.append(wt[j])
            if df.loc[j, 'treated']==0:
                cn.append((df.loc[j, 'outcome'])*wt[j])
                cnw.append(wt[j])   
        cna=sum(cn)/sum(cnw)
        tra=sum(tr)/sum(trw)
        CATEcate.append(tra-cna)
    return CATEcate
        
CATEcate=fakeCATE2(m)
print (CATEcate)

#%%
def fakeCATE1(pym, treatment_column='treated', outcome_column='outcome'):
    CATElist=[]
    ans=[]
    for i in ordint:
        wt=MMG_dictall[i][1]
        tr=[]
        cn=[]
        for j in wt.index.values:
            if df.loc[j, 'treated']==1:
                tr.append(df.loc[j, 'outcome'])
            if df.loc[j, 'treated']==0:
                cn.append(df.loc[j, 'outcome'])
        cna=sum(cn)/len(cn)
        tra=sum(tr)/len(tr)
        CATElist.append(tra-cna)
    return CATElist

CATElist=fakeCATE1(m)
print (CATElist)

###THIS HAS A PROBLEM
#%%
def mmg_of_unit(pym, index):
    return MMG_dict[index]

'''
print (mmg_of_unit(m, 5))
'''

#%%
#THIS IS REDUNDANT
def MG_internal(pym):

    k=set()
    mmg_dict = {}
    for i in ordint:
        mmg = mmg_of_unit(m, i)
        if type(mmg) != bool:
            r=hash(mmg.values.tobytes())
            if r not in k:
                k.add(r)
                mmg_dict[i]=mmg
    mmg_dict = dict(enumerate(mmg_dict[x] for x in sorted(mmg_dict)))
    return mmg_dict

MG_dict=MG_internal(m)
'''
print (MG_dict[0])
'''
#%%
def unit_weights(pym):
    weights = [0] * len(ordint)
    for i in range(len(ordint)):
        for j in MMG_dict[i].index:
            weights[j] += 1
    return weights

pym_weights = unit_weights(m)
'''
print (pym_weights)
'''
#%%This is same as ATE defined as vol0 for FLAME, and given in MALTS documentation.
def ATE_v0(pym):
    return pym.CATE_df['avg.CATE'].mean()

print (ATE_v0(m))

#%%
def te_of_unit(pym, index):
    return pym.CATE_df.loc[index, 'avg.CATE']

print (te_of_unit(m, 1))
 
#%%
CATElist=m.CATE_df['avg.CATE']
#%%
print (pym_weights)

#%%
def ATTwrong(pym):
    te_list=[]
    for i in ordint:
        if df.loc[i, 'treated']==1:
            MGi=pym.MG_matrix.loc[i]
            MGi=MGi[MGi>0]
            entries=MGi.index.values
            group=df.loc[entries]
            nul=group[group['treated']==0]
            nulwt=0
            nulsmwt=0
            for ind, row in nul.iterrows():
                nulsmwt=nulsmwt+MGi[ind]*nul.loc[ind, 'outcome']
                nulwt=nulwt+MGi[ind]
            nulavg=nulsmwt/nulwt
            x=df.loc[i, 'outcome']-nulavg
            te_list.append(x)
    return sum(te_list)/len(te_list)

print (ATT(m))
#%%






