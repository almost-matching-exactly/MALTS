# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:21 2020

@author: Harsh
"""
import pandas as pd
import numpy as np
import pymalts
import warnings
warnings.filterwarnings("ignore")

nsw = pd.read_stata('http://www.nber.org/~rdehejia/data/nsw.dta')
psid_control = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls.dta')
psid_control2 = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls2.dta')
psid_control3 = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls3.dta')

nsw = nsw.drop(columns=['data_id','re74'],errors='ignore')
psid_control = psid_control.drop(columns=['data_id','re74'],errors='ignore')
psid_control2 = psid_control2.drop(columns=['data_id','re74'],errors='ignore')
psid_control3 = psid_control3.drop(columns=['data_id','re74'],errors='ignore')

data = nsw.append(psid_control2,ignore_index=True)

m = pymalts.malts_mf('re78', 'treat', data=nsw,
                     discrete=['black','hispanic','married','nodegree'],n_splits=2)

cate_df = m.CATE_df['CATE']
cate_df['avg.CATE'] = cate_df.mean(axis=1)
cate_df['std.CATE'] = cate_df.std(axis=1)
cate_df['re78'] = m.CATE_df['outcome'].mean(axis=1)
cate_df['treat'] = m.CATE_df['treatment'].mean(axis=1)
print(np.mean(cate_df['avg.CATE']))
m_opt_list = m.M_opt_list
