
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed

from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sqlalchemy import create_engine

import matplotlib.pyplot as plt


def construct_sec_order(arr):
    # data generation function helper.
    
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)

def data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant, 
                            control_m = 0.1, treated_m = 0.9):
    # the data generation function that I'll use.
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.05, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.05, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]

    yc = np.dot(xc, np.array(dense_bs)) + errors1     # y for conum_treatedrol group 
    
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec + errors2    # y for treated group 

    xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   #
    xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   #
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns=['{0}'.format(i) for i in range(num_cov_dense + num_covs_unimportant)])
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns=['{0}'.format(i) for i in range(num_cov_dense + num_covs_unimportant )] ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef


# In[4]:

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres = 0, tradeoff = 0.1):
    
    covs_to_match_on = set(cov_l) - {c} # the covariates to match on
    
    # the flowing query fetches the matched results (the variates, the outcome, the treatment indicator)
    s = time.time()
    ##cur.execute('''with temp AS 
    ##    (SELECT 
    ##    {0}
    ##    FROM {3}
    ##    where "matched"=0
    ##    group by {0}
    ##    Having sum("treated")>'0' and sum("treated")<count(*) 
    ##    )
    ##    (SELECT {1}, {3}."treated", {3}."outcome"
    ##    FROM temp, {3}
    ##    WHERE {2}
    ##    )
    ##    '''.format(','.join(['"C{0}"'.format(v) for v in covs_to_match_on ]),
    ##               ','.join(['{1}."C{0}"'.format(v, db_name) for v in covs_to_match_on ]),
    ##               ' AND '.join([ '{1}."C{0}"=temp."C{0}"'.format(v, db_name) for v in covs_to_match_on ]),
    ##               db_name
    ##              ) )
    ##res = np.array(cur.fetchall())
    
    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("treated")>'0' and sum("treated")<count(*) 
        )
        (SELECT {1}, treated, outcome
        FROM {3}
        WHERE EXISTS 
        (SELECT 1
        FROM temp 
        WHERE {2}
        )
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_to_match_on ]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   db_name
                  ) )
    res = np.array(cur.fetchall())
    
    time_match = time.time() - s
    
    s = time.time()
    # the number of unmatched treated units
    cur.execute('''select count(*) from {} where "matched" = 0 and "treated" = 0'''.format(db_name))
    num_control = cur.fetchall()
    # the number of unmatched control units
    cur.execute('''select count(*) from {} where "matched" = 0 and "treated" = 1'''.format(db_name))
    num_treated = cur.fetchall()
    time_BF = time.time() - s
    
    # fetch from database the holdout set
    
    ##s = time.time()
    ##cur.execute('''select {0}, "treated", "outcome" 
    ##               from {1}
    ##            '''.format( ','.join( ['"C{0}"'.format(v) for v in covs_to_match_on ] ) , holdout))
    ##holdout = np.array(cur.fetchall())
    
    s = time.time() # the time for fetching data into memory is not counted if use this
    
    # below is the regression part for PE
    #ridge_c = Ridge(alpha=0.1)
    #ridge_t = Ridge(alpha=0.1)
    tree_c = DecisionTreeRegressor(max_depth=8, random_state=0)
    tree_t = DecisionTreeRegressor(max_depth=8, random_state=0)
    
    holdout = holdout_df.copy()
    holdout = holdout[ ["{}".format(c) for c in covs_to_match_on] + ['treated', 'outcome']]
    
    mse_t = np.mean(cross_val_score(tree_t, holdout[holdout['treated'] == 1].iloc[:,:-2], 
                                holdout[holdout['treated'] == 1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    mse_c = np.mean(cross_val_score(tree_c, holdout[holdout['treated'] == 0].iloc[:,:-2], 
                                holdout[holdout['treated'] == 0]['outcome'], scoring = 'neg_mean_squared_error' ) )
    
    #mse_t = np.mean(cross_val_score(ridge_t, holdout[holdout['treated'] == 1].iloc[:,:-2], 
    #                            holdout[holdout['treated'] == 1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    #mse_c = np.mean(cross_val_score(ridge_c, holdout[holdout['treated'] == 0].iloc[:,:-2], 
    #                            holdout[holdout['treated'] == 0]['outcome'], scoring = 'neg_mean_squared_error' ) )
    # above is the regression part for BF
    
    time_PE = time.time() - s
    
    if len(res) == 0:
        return (( mse_t + mse_c ), time_match, time_PE, time_BF)
        ##return mse_t + mse_c
    else:        
        return (tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + float(len(res[res[:,-2]==1]))/num_treated[0][0]) +             ( mse_t + mse_c ), time_match, time_PE, time_BF)
        ##return reg_param * (float(len(res[res[:,-2]==0]))/num_control[0][0] + float(len(res[res[:,-2]==1]))/num_treated[0][0]) +\
        ##         ( mse_t + mse_c )


# In[5]:

# update matched units
# this function takes the currcent set of covariates and the name of the database; and update the "matched"
# column of the newly mathced units to be "1"

def update_matched(covs_matched_on, db_name, level):
    
    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("treated")>'0' and sum("treated")<count(*) 
        )
        update {3} set "matched"={4}
        WHERE EXISTS
        (SELECT {0}
        FROM temp
        WHERE {2} and {3}."matched" = 0
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_matched_on]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_matched_on ]),
                   db_name,
                   level
                  ) )
    conn.commit()
    
    return


# In[6]:

# get CATEs 
# this function takes a list of covariates and the name of the data table as input and outputs a dataframe 
# containing the combination of covariate values and the corresponding CATE 
# and the corresponding effect (and the count and variance) as values

def get_CATE_db(cov_l, db_name, level):

    cur.execute(''' select {0}, avg(outcome * 1.0), count(*)
                    from {1}
                    where matched = {2} and treated = 0
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_c = cur.fetchall()
    
    cur.execute(''' select {0}, avg(outcome * 1.0), count(*)
                    from {1}
                    where matched = {2} and treated = 1
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_t = cur.fetchall()
            
    if (len(res_c) == 0) | (len(res_t) == 0):
        return None
    
    cov_l = list(cov_l)
    
    result = pd.merge(pd.DataFrame(np.array(res_c), columns=['{}'.format(i) for i in cov_l]+['effect_c', 'count_c']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['effect_t', 'count_t']), 
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
    result_df = result[['{}'.format(i) for i in cov_l] + ['effect_c', 'effect_t', 'count_c', 'count_t']]
        
    # -- the following section are moved to after getting the result
    #d = {}
    #for i, row in result.iterrows():
    #    k = ()
    #    for j in range(len(cov_l)):
    #        k = k + ((cov_l[j], row[j]),)
    #    d[k] = (row['effect_c'], row['effect_t'], row['std_t'], row['std_c'], row['count_c'], row['count_t'])
    # -- the above section are moved to after getting the result
    
    return result_df


# In[7]:

def run_db(db_name, holdout_df, num_covs, reg_param = 0.1):

    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    covs_dropped = [] # covariate dropped
    ds = []
    
    level = 1

    timings = [0]*5 # first entry - match (groupby and join), 
                    # second entry - regression (compute PE), 
                    # third entry - compute BF, 
                    # fourth entry - keep track of CATE, 
                    # fifth entry - update database table (mark matched units). 
    
    cur_covs = list(range(num_covs)) # initialize the current covariates to be all covariates
        
    # make predictions and save to disk
    s = time.time()
    update_matched(cur_covs, db_name, level) # match without dropping anything
    timings[4] = timings[4] + time.time() - s
        
    s = time.time()
    d = get_CATE_db(cur_covs, db_name, level) # get CATE without dropping anything
    timings[3] = timings[3] + time.time() - s
    
    ds.append(d)
    
    ##s = time.time()
    ##cur.execute('''update {} set "matched"=2 WHERE "matched"=1 '''.format(db_name)) # mark the matched units as matched and 
                                                                                    #they are no langer seen by the algorithm
    ##conn.commit()
    ##timings[4] = timings[4] + time.time() - s
    
    while len(cur_covs)>1:
        
        print(cur_covs) # print current set of covariates
        
        level += 1
        
        # the early stopping conditions
        cur.execute('''select count(*) from {} where "matched"=0 and "treated"=0'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            break
        cur.execute('''select count(*) from {} where "matched"=0 and "treated"=1'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            break
        
        best_score = -np.inf
        cov_to_drop = None
        
        cur_covs = list(cur_covs)
        for c in cur_covs:
            
            score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                      holdout_df, tradeoff = 0.1)
            
            timings[0] = timings[0] + time_match
            timings[1] = timings[1] + time_PE
            timings[2] = timings[2] + time_BF
            if score > best_score:
                best_score = score
                cov_to_drop = c

        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set
        
        s = time.time()
        update_matched(cur_covs, db_name, level)
        timings[4] = timings[4] + time.time() - s
        
        s = time.time()
        d = get_CATE_db(cur_covs, db_name, level)
        timings[3] = timings[3] + time.time() - s
        
        ds.append(d)
        
        ##s = time.time()
        ##cur.execute('''update {} set "matched"=2 WHERE "matched"=1 '''.format(db_name))
        ##conn.commit()
        ##timings[4] = timings[4] + time.time() - s
        
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate
        
    return timings, ds


# In[2]:

if __name__ == '__main__':


    # call the 'run' function to run the algorithm. the first argument is the name datatable of interest, 
    # the second argument is the holdout dataframe, the third argument is the number of covariates,
    # and the fourth argument (optional) is the trade-off parameter. Larger tradeoff parameter puts more weight on matching more data, 
    # while smaller tradeoff parameter puts more weight on predicting the result correctly. 
    # the result is saved in a pickle file named as "FLAME-bit-result"

    #### IMPORTANT NOTE: The datatable of interest is stored as an Microsoft databse table while the holdout dataset is store in memory as a Pandas dataframe. 

    ## --below is an example


    conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=localhost;"
                        "Database=master;"
                        "Trusted_Connection=yes;")

    cur = conn.cursor()

    engine = create_engine('mssql+pyodbc://localhost/master?driver=SQL+Server+Native+Client+11.0')

    df,betas,eff_coef = data_generation_dense_2(10000, 10000, 10, 5)
    holdout_df,_,_ = data_generation_dense_2(10000, 10000, 10, 5)
    db_name = 'db'
    cur.execute('drop table if exists {}'.format(db_name))
    conn.commit()

    df.to_sql(db_name, engine)
        
    res = run(db_name, holdout_df, 15)
    pickle.dump(res, open('FLAME-db-result', 'wb'))
    ## --above is an example