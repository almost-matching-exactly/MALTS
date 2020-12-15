import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import scipy
np.random.seed(0)
import pymalts_test as pym
import dame_flame as dmflm
import time
import pandas as pd
#%%
concolumnlist=dict()
for i in range(100):
    x='C'+str(i+1)
    concolumnlist[i]=x
#%%
discolumnlist=dict()
for i in range(100):
    x='D'+str(i+1)
    discolumnlist[i]=x
#%%
def mixdatasample(n_unit, n_con, n_dis, categ=2):
    dg=np.random.randn(n_unit,n_con)
    dgf=pd.DataFrame(dg)
    dgf.rename(columns=lambda s: concolumnlist[s], inplace=True)
    dh=np.random.randint(categ, size=(n_unit,n_dis))
    dhf=pd.DataFrame(dh)    
    dhf.rename(columns=lambda s: discolumnlist[s], inplace=True)
    di=np.random.randint(2, size=n_unit)
    dif=pd.DataFrame(di)
    dif.rename(columns={0:'treated'}, inplace=True)
    dj=np.random.randn(n_unit,1)
    djf=pd.DataFrame(dj)
    djf.rename(columns={0:'outcome'}, inplace=True)
    dk=pd.concat([dgf,dhf,dif,djf], axis=1)
    return dk, list(dhf.columns)

#%%
dis_con_dict=dict()
list_dis=list()
list_con=list()
for i in range(3, 19):
    list_dis.append(str(i)+' dis')
    list_con.append(str(i)+' con')
#%%
for i in list_dis:
    dis_con_dict[i]=[0.0 for i in range(len(list_con))]    
dis_con_matrix=pd.DataFrame(dis_con_dict,index=list_con)    
#%%   
'''400 units, n_splits=2, categ=2, vary continuous and discrete units from 3 to 18''' 
for i in range(3,19):
    for j in range(3,19):
        for k in range(3):
            dataframe, discretelist = mixdatasample(400,i,j)
            x=time.time()
            m=pym.malts_mf(outcome='outcome', treatment='treated', discrete=discretelist, data=dataframe, n_splits=2)
            y=time.time()
            dis_con_matrix[str(j)+' dis'][str(i)+' con']+=y-x
            print (" trial ",(k+1)," i ",i," j ",j," time ",y-x)
        dis_con_matrix[str(j)+' dis'][str(i)+' con']/=3   
#%%
dis_con_matrix.to_excel('18dis13con.xlsx')
#%%
plt.pcolor(dis_con_matrix)
#%%
plt.subplots(figsize=(20,15))
ax = sns.heatmap(dis_con_matrix, cmap = "Greens", xticklabels=True, yticklabels=True)

#%%