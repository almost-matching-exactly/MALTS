
# coding: utf-8

# In[1]:


library('MatchIt')


# In[9]:


library('grf') # the random forest package


# In[42]:


mydata <- read.csv('densecore_df.csv')


# In[20]:


X <- sapply(mydata[,1:30], as.numeric)

tau.forest = causal_forest(X, mydata$outcome, mydata$treated)   # train the model 
tau.hat = predict(tau.forest, X[10001:20000,]) # predict on the treated group, which is the second half of the data table.
# the above two lines are from the official github.


# In[25]:


tau.hat$predict # this is the predicted result


# In[29]:


res = X[10001:20000,] # this is the second half of the data table, which is the treated group


# In[31]:


res$pred = tau.hat$predict # align the predictions with the labels


# In[43]:


# genmatch, 1-PSNNM, Mahalanobis all come from here.
# with different values for 'method' parameter of the 'matchit' function, 
# this 'matchit' function performs different matching.

# method = 'genetic' -- genmatch
# method = 'Mahalanobis' -- Mahalanobis
# method = 'logistic' -- 1-PSNNM
# method = 'Mahalanobis' -- Mahalanobis


r <- matchit(treated ~ X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29,
             methods = 'genetic', data = mydata)

# following line: after matching, compute the causal effect by subtraction. 
hh <- mydata[as.numeric(names(r$match.matrix[,])),'outcome']- mydata[as.numeric(r$match.matrix[,]),'outcome']


# In[44]:


mydata[10001:20000,'pred'] = hh

