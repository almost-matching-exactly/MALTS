#%% Imports
import time
import pandas as pd
import pymalts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import scipy
np.random.seed(0)
df = pd.read_csv("C:/Users/angik/ANGIKAR/ANGIKAR_AME/CynthiaRudin_AME_Project/Additional_Functions/Visu&TimeMalts/example.csv", index_col=0)
#%%NEW FROM HERE
columnlist=dict()
for i in range(100):
    x='X'+str(i+1)
    columnlist[i]=x
#%%
dk=datasample(2500,18)
dff = df.sort_index(ascending=True)
print(dk.head())
print(dff.head())
print(df.head())
#%%
def datasample(n_unit, n_cov):
    dg=np.random.randn(n_unit,n_cov)
    dh=np.random.randint(2, size=n_unit)
    dj=np.random.randn(n_unit,1)
    dgf=pd.DataFrame(dg)
    dgf.rename(columns=lambda s: columnlist[s], inplace=True)
    dhf=pd.DataFrame(dh)
    dhf.rename(columns={0:'treated'}, inplace=True)
    djf=pd.DataFrame(dj)
    djf.rename(columns={0:'outcome'}, inplace=True)
    dk=pd.concat([dgf,djf,dhf], axis=1)
    return dk
#%%
dictunit=dict()
for i in range(10,27):
    dq=datasample(i*50,3)
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print((50*i), " units with 2 folds and 3 covariates takes ", (y-x), " time.")
    dictunit[50*i]=(y-x)
#%%
dictunit2=dict()
for i in range(10,27):
    dq=datasample(i*50,3)
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print((50*i), " units with 2 folds and 3 covariates takes ", (y-x), " time.")
    dictunit2[50*i]=(y-x)
#%%
subsetdict=dict()
for i in range(10,27):
    dq=dff.loc[0:(50*i-1)]
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print((50*i), " units with 2 folds and 18 covariates takes ", (y-x), " time.")
    subsetdict[50*i]=(y-x)
#%%
newdictunit=dict()
for i in range(10,27):
    dq=datasample(i*50,18)
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print((50*i), " units with 2 folds and 18 covariates takes ", (y-x), " time.")
    newdictunit[50*i]=(y-x)
#%%
otherdictunit=dict()
for i in range(3, 21):
    dq=datasample(1000,i)
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print(1000, " units with 2 folds and ", i, " covariates takes ", (y-x), " time.")
    otherdictunit[i]=(y-x)
#%%
otherdictunit2=dict()
for i in range(3, 21):
    dq=datasample(1000,i)
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dq, n_splits=2)
    y=time.time()
    print(1000, " units with 2 folds and ", i, " covariates takes ", (y-x), " time.")
    otherdictunit2[i]=(y-x)
#%%
for i in range(10,27):
    print(50*i, " - Real data: ", subsetdict[50*i], ", Fake data: ", newdictunit[50*i])
#%%LinRegPNG
a=plt.scatter(newdictunit.keys(), newdictunit.values(), c='r')
b=plt.scatter(subsetdict.keys(), subsetdict.values(), c='b')
plt.xlabel('Number of Units in Dataframe')
plt.ylabel('Time taken for Malts to run')

m1, b1 = np.polyfit(list(newdictunit.keys()), list(newdictunit.values()), 1)
print (m1, b1)
x=np.array(list(newdictunit.keys()))
plt.plot(x, m1*x+b1, c='r')

m2, b2 = np.polyfit(list(subsetdict.keys()), list(subsetdict.values()), 1)
print (m2, b2)
x=np.array(list(subsetdict.keys()))
plt.plot(x, m2*x+b2, c='b')

plt.ylim(0,140)
plt.legend((a,b), ('Random Data', 'Actual Data'),  scatterpoints=1, loc='best', fontsize=8)
plt.show()
#%%
plt.plot(list(newdictunit.keys()), list(newdictunit.values()))
#%%
orddfdict=dict()
for i in range(2,6):
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=dff, n_splits=i)
    y=time.time()
    print(i, " splits take ", (y-x))
    orddfdict[i]=(y-x)
print (orddfdict)
#%%
dfdict=dict()
for i in range(2,6):
    x=time.time()
    m=pymalts.malts_mf( outcome='outcome', treatment='treated', data=df, n_splits=i)
    y=time.time()
    print(i, " splits take ", (y-x))
    dfdict[i]=(y-x)
print (dfdict)
#%%
onlytrygbrlist=dfdict.copy()
plt.bar(range(len(onlytrygbrlist)), onlytrygbrlist.values(), align='center')
plt.xticks(range(len(onlytrygbrlist)), list(onlytrygbrlist.keys()))
plt.xlabel('Number of splits of Multifold Malts')
plt.ylabel('Time taken in seconds')
plt.show()
#%%
onlytrygbrlistscale=dict()
for i in onlytrygbrlist.keys():
    onlytrygbrlistscale[i]=onlytrygbrlist[i]/i
#%%
plt.plot(list(onlytrygbrlist.keys()), list(onlytrygbrlist.values()), marker='*')
plt.xlabel('Number of splits of Multifold Malts')
plt.xticks(range(min(onlytrygbrlist.keys()),max(onlytrygbrlist.keys())+1), list(onlytrygbrlist.keys()))
plt.ylim(100,600)
plt.ylabel('Time taken in seconds')
plt.show()
#%%
a=plt.scatter(orddfdict.keys(), orddfdict.values(), c='r')
b=plt.scatter(dfdict.keys(), dfdict.values(), c='b')
plt.xlabel('Number of splits of Multifold Malts')
plt.ylabel('Time taken in seconds')
plt.ylim(0,600)
m1, b1 = np.polyfit(list(orddfdict.keys()), list(orddfdict.values()), 1)
print (m1, b1)
x=np.array(list(orddfdict.keys()))
plt.plot(x, m1*x+b1, c='r')

m2, b2 = np.polyfit(list(dfdict.keys()), list(dfdict.values()), 1)
print (m2, b2)
x=np.array(list(dfdict.keys()))
plt.plot(x, m2*x+b2, c='b')
plt.xticks(range(min(onlytrygbrlist.keys()),max(onlytrygbrlist.keys())+1), list(onlytrygbrlist.keys()))
plt.legend((a,b), ('Ordered Dataframe', 'Unordered Dataframe'),  scatterpoints=1, loc='best', fontsize=8)
plt.show()
#%%
a=plt.scatter(dictunit.keys(), dictunit.values(), c='r')
b=plt.scatter(dictunit2.keys(), dictunit2.values(), c='b')
plt.xlabel('Number of units in DataFrame, n_fold=2')
plt.ylabel('Time taken in seconds')
plt.ylim(0,100)

m1, b1 = np.polyfit(list(dictunit.keys()), list(dictunit.values()), 1)
print (m1, b1)
x=np.array(list(dictunit.keys()))
plt.plot(x, m1*x+b1, c='r')

m2, b2 = np.polyfit(list(dictunit2.keys()), list(dictunit2.values()), 1)
print (m2, b2)
x=np.array(list(dictunit2.keys()))
plt.plot(x, m2*x+b2, c='b')

plt.xlim(400,2100)
plt.legend((a,b), ('Trial 1', 'Trial 2'),  scatterpoints=1, loc='best', fontsize=8)
plt.show()
#%%
a=plt.scatter(otherdictunit.keys(), otherdictunit.values(), c='r')
b=plt.scatter(otherdictunit2.keys(), otherdictunit2.values(), c='b')
plt.xlabel('Number of covariates in DataFrame, n_fold=2')
plt.ylabel('Time taken in seconds')
plt.ylim(0,100)

m1, b1 = np.polyfit(list(otherdictunit.keys()), list(otherdictunit.values()), 1)
print (m1, b1)
x=np.array(list(otherdictunit.keys()))
plt.plot(x, m1*x+b1, c='r')

m2, b2 = np.polyfit(list(otherdictunit2.keys()), list(otherdictunit2.values()), 1)
print (m2, b2)
x=np.array(list(otherdictunit2.keys()))
plt.plot(x, m2*x+b2, c='b')

plt.xticks(range(min(otherdictunit2.keys()),max(otherdictunit2.keys())+1), list(otherdictunit2.keys()))
plt.legend((a,b), ('Trial 1', 'Trial 2'),  scatterpoints=1, loc='best', fontsize=8)
plt.show()
#%%
m5 = pymalts.malts_mf( outcome='outcome', treatment='treated', data=df, n_splits=5)
#%%
def plotmatchindex(mlts, datfram, ind, var1, var2):
    MG0 = mlts.MG_matrix.loc[ind]
    matched_units_idx = MG0[MG0!=0].index 
    matched_units = datfram.loc[matched_units_idx]
    sns.lmplot(x=var1, y=var2, hue='treated', data=matched_units,palette="Set1")
    plt.scatter(x=[datfram.loc[ind,var1]],y=[datfram.loc[ind,var2]],c='black',s=100)
    string='Matched Group for Unit-'+str(ind)
    plt.title(string)
    
plotmatchindex(m5, df, 0, 'X1', 'X2')
#%%
def plotavgcatecovar(mlts, datfram, var):
    data_w_cate=pd.concat([datfram, mlts.CATE_df], axis=1)
    data_w_cate = data_w_cate.drop(columns=['outcome','treated']) 
    sns.regplot( x=var, y='avg.CATE', data=data_w_cate, scatter_kws={'alpha':0.5,'s':2}, line_kws={'color':'black'}, order=2 )
    print (data_w_cate.head())
plotavgcatecovar(m5,df,'X1')
#%%
def plotstdcatecovar(mlts, datfram, var):
    data_w_cate=pd.concat([datfram, mlts.CATE_df], axis=1)
    data_w_cate = data_w_cate.drop(columns=['outcome','treated']) 
    sns.regplot( x=var, y='std.CATE', data=data_w_cate, scatter_kws={'alpha':0.5,'s':2}, line_kws={'color':'black'}, order=2 )
    print (data_w_cate.head())
plotstdcatecovar(m5,df,'X1')
#%%
def whoismax(mlts, datfram, ind):
    MG0 = mlts.MG_matrix.loc[ind]
    matched_units_idx = MG0[MG0!=0].index 
    matched_units = datfram.loc[matched_units_idx]
    return (pd.DataFrame(matched_units))
print (whoismax(m5,df,1))
#%%
netmmgarray=[]
for i in sorted(df.index):
    if (type(whoismax(m5,df,i))==pd.core.frame.DataFrame):
        netmmgarray.append(len(whoismax(m5,df,i)))
    else:
        netmmgarray.append(0)
print (netmmgarray)
#%%
countermmgarray=dict()
for i in netmmgarray:
    if i not in countermmgarray.keys():
        countermmgarray[i]=0
    countermmgarray[i]+=1

print(countermmgarray)
#%%
plt.scatter(countermmgarray.keys(), countermmgarray.values(), c='r')
plt.xlabel('Number of Units in Main Matched Group (Unweighted)')
plt.ylabel('Number of Such Units')
plt.show()
#%%
def mgindexplot(mlts, datfram, ind):
    v=mlts.MG_matrix.loc[ind, list(whoismax(mlts,datfram,ind).index)]
    indexdict=dict()
    for i, j in v.iteritems():
        k=int(j)
        if k not in indexdict.keys():
            indexdict[k]=0
        indexdict[k]+=1
    plt.bar(range(len(indexdict)), indexdict.values(), align='center', color='g')
    plt.xticks(range(len(indexdict)), list(indexdict.keys()))
    plt.xlabel('Weight of units')
    string='Number of such units for MG of unit ' + str(ind)
    plt.ylabel(string)
    plt.show()
mgindexplot(m5,df,3)
#%%
covlist=list(df.columns)
covlist.remove('treated')
covlist.remove('outcome')
#%%
def comparemg(mlts,datfram,ind):
    zitem=dict()
    oitem=dict()
    zw=0
    ow=0
    v=mlts.MG_matrix.loc[ind, list(whoismax(mlts,datfram,ind).index)]
    for ix, wt in v.iteritems():
        if datfram.loc[ix, 'treated']==1:
            ow=ow+mlts.MG_matrix.loc[ind, ix]
        else:
            zw=zw+mlts.MG_matrix.loc[ind, ix]
    for i in covlist:
        zitem[i]=0
        oitem[i]=0
        for ix, wt in v.iteritems():
            if datfram.loc[ix, 'treated']==1:
                oitem[i]=oitem[i]+(mlts.MG_matrix.loc[ind, ix]*datfram.loc[ix, i])
            else:
                zitem[i]=zitem[i]+(mlts.MG_matrix.loc[ind, ix]*datfram.loc[ix, i])
    for i in covlist:
        zitem[i]=zitem[i]/zw
        oitem[i]=oitem[i]/ow
    a=plt.scatter(zitem.keys(), zitem.values(), c='r')
    b=plt.scatter(oitem.keys(), oitem.values(), c='b')
    for i in range(19):
        plt.axvline(x=i-0.5, color='black',linestyle='--')
    plt.xlabel('Covariate')
    plt.ylabel('Treated and Control Unit Values')
    string='Matched Group for Index '+str(ind)
    plt.title(string)
    plt.legend((a,b), ('Control Units', 'Treated Units'),  scatterpoints=1, loc='best', fontsize=8)
comparemg(m5,df,0)
#%%
def whoismaxmore(mlts, datfram, ind):
    MG0 = mlts.MG_matrix.loc[ind]
    matched_units_idx = MG0[MG0>2].index 
    matched_units = datfram.loc[matched_units_idx]
    return (pd.DataFrame(matched_units))
#%%











































