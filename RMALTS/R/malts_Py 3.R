self.epsilon = epsilon #lambda x: (1 + np.exp( - self.epsilon) )/(1+np.exp( self.gamma * (x - self.epsilon) ) )

n <- nrow(data)
p <- ncol(data)-2 # self.n, self.p = data.shape # self.p = self.p - 2

continuous <- colnames(data)-c(outcome, treatment, discrete) # self.continuous = set(data.columns).difference(set([outcome]+[treatment]+discrete))
# #        Mc = np.ones((len(self.continuous),)) #initializing the stretch vector
# #        Md = np.ones((len(self.discrete),)) #initializing the stretch vector

# #splitting the data into control and treated units
df_T <- # self.df_T = data.loc[data[treatment]==1]
df_C <- # self.df_C = data.loc[data[treatment]==0]
# #extracting relevant covariates (discrete,continuous)
# #and outcome. Converting to numpy array.
Xc_T <- # self.Xc_T = self.df_T[self.continuous].to_numpy()
Xc_C <- # self.Xc_C = self.df_C[self.continuous].to_numpy()
Xd_T <- # self.Xd_T = self.df_T[self.discrete].to_numpy()
Xd_C <- # self.Xd_C = self.df_C[self.discrete].to_numpy()
Y_T <- # self.Y_T = self.df_T[self.outcome].to_numpy()
Y_C <- # self.Y_C = self.df_C[self.outcome].to_numpy()
del2_Y_T <- # self.del2_Y_T = ((np.ones((len(self.Y_T),len(self.Y_T)))*self.Y_T).T - (np.ones((len(self.Y_T),len(self.Y_T)))*self.Y_T))**2
del2_Y_C <- # self.del2_Y_C = ((np.ones((len(self.Y_C),len(self.Y_C)))*self.Y_C).T - (np.ones((len(self.Y_C),len(self.Y_C)))*self.Y_C))**2
#
Dc_T <- # self.Dc_T = np.ones((self.Xc_T.shape[0],self.Xc_T.shape[1],self.Xc_T.shape[0])) * self.Xc_T.T
Dc_T <- # self.Dc_T = (self.Dc_T - self.Dc_T.T)
Dc_C <- # self.Dc_C = np.ones((self.Xc_C.shape[0],self.Xc_C.shape[1],self.Xc_C.shape[0])) * self.Xc_C.T
Dc_C <- # self.Dc_C = (self.Dc_C - self.Dc_C.T)
#
Dd_T <- # self.Dd_T = np.ones((self.Xd_T.shape[0],self.Xd_T.shape[1],self.Xd_T.shape[0])) * self.Xd_T.T
Dd_T <- # self.Dd_T = (self.Dd_T != self.Dd_T.T)
Dd_C <- # self.Dd_C = np.ones((self.Xd_C.shape[0],self.Xd_C.shape[1],self.Xd_C.shape[0])) * self.Xd_C.T
Dd_C <- # self.Dd_C = (self.Dd_C != self.Dd_C.T)


threshold <- function(self,x){
  #   k = self.k
  # for i in range(x.shape[0]):
  #   row = x[i,:]
  # row1 = np.where( row < row[np.argpartition(row,k+1)[k+1]],1,0)
  # x[i,:] = row1
  # return x
}


distance <- function(self,Mc,Md,xc1,xd1,xc2,xd2){
  #   dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
  # dd = np.sum((Md**2)*xd1!=xd2)
  # return dc+dd
  #
  # def loss_(self, Mc, Md, xc1, xd1, y1, xc2, xd2, y2, gamma=1 ):
  #   w12 = np.exp( -1 * gamma * self.distance(Mc,Md,xc1,xd1,xc2,xd2) )
  # return w12*((y1-y2)**2)
}


calcW_T <- function(self,Mc,Md){
  #   #this step is slow
  Dc <- #   Dc = np.sum( ( self.Dc_T * (Mc.reshape(-1,1)) )**2, axis=1)
  Dd <- # Dd = np.sum( ( self.Dd_T * (Md.reshape(-1,1)) )**2, axis=1)
  W <- # W = self.threshold( (Dc + Dd) )
  W <- # W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
  return(W) # return W
}


calcW_C <- function(self,Mc,Md){
  #   #this step is slow
  Dc <- #   Dc = np.sum( ( self.Dc_C * (Mc.reshape(-1,1)) )**2, axis=1)
  Dd <- # Dd = np.sum( ( self.Dd_C * (Md.reshape(-1,1)) )**2, axis=1)
  W <- # W = self.threshold( (Dc + Dd) )
  W <- # W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
  return(W) # return W
}

Delta_ <- function(self,Mc,Md){
  W_T <- # self.W_T = self.calcW_T(Mc,Md)
  W_C <- # self.W_C = self.calcW_C(Mc,Md)
  delta_T <- # self.delta_T = np.sum((self.Y_T - (np.matmul(self.W_T,self.Y_T) - np.diag(self.W_T)*self.Y_T))**2)
  delta_C <- # self.delta_C = np.sum((self.Y_C - (np.matmul(self.W_C,self.Y_C) - np.diag(self.W_C)*self.Y_C))**2)
  return(delta_T + delta_C) # return self.delta_T + self.delta_C
}


objective <- function(M){
  # self.continuous = set(data.columns).difference(set([outcome]+[treatment]+discrete))
  Mc <- subset(M,len(continuous)) # Mc = M[:len(self.continuous)]
  Md <- M[len(continuous):] # Md = M[len(self.continuous):]
  delta <- self.Delta_(Mc,Md) # delta = self.Delta_(Mc,Md)
  reg <- self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 ) # reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 )
  cons1 <- 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2 # cons1 = 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2
  cons2 <- 1e+25 * np.sum( ( np.concatenate((Mc, Md)) < 0 ) ) # cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md)) < 0 ) )
  return(delta + reg + cons1 + cons2) # return delta + reg + cons1 + cons2
}


fit <- function(method ='BFGS', data, objective){
  p <- ncol(data)-2
  M_init <- matrix(1,p,1)  #   M_init = np.ones((self.p,))
  res <- optim(M_init, objective, method = method) # res <- opt.minimize( self.objective, x0=M_init,method=method )
  M <- res.par # M = res.x
  # Mc <- self.M[:len(self.continuous)]
  # Md <- self.M[len(self.continuous):]
  return(res) # return res
}



# def get_matched_groups(self, df_estimation, k=10 ):
#   #units to be matched
#   Xc = df_estimation[self.continuous].to_numpy()
# Xd = df_estimation[self.discrete].to_numpy()
# Y = df_estimation[self.outcome].to_numpy()
# T = df_estimation[self.treatment].to_numpy()
# #splitted estimation data for matching
# df_T = df_estimation.loc[df_estimation[self.treatment]==1]
# df_C = df_estimation.loc[df_estimation[self.treatment]==0]
# #converting to numpy array
# Xc_T = df_T[self.continuous].to_numpy()
# Xc_C = df_C[self.continuous].to_numpy()
# Xd_T = df_T[self.discrete].to_numpy()
# Xd_C = df_C[self.discrete].to_numpy()
# Y_T = df_T[self.outcome].to_numpy()
# Y_C = df_C[self.outcome].to_numpy()
# D_T = np.zeros((Y.shape[0],Y_T.shape[0]))
# D_C = np.zeros((Y.shape[0],Y_C.shape[0]))
# #distance_treated
# Dc_T = (np.ones((Xc_T.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_T.shape[0])) * Xc_T.T).T)
# Dc_T = np.sum( (Dc_T * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
# Dd_T = (np.ones((Xd_T.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_T.shape[0])) * Xd_T.T).T )
# Dd_T = np.sum( (Dd_T * (self.Md.reshape(-1,1)) )**2 , axis=1 )
# D_T = (Dc_T + Dd_T).T
# #distance_control
# Dc_C = (np.ones((Xc_C.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_C.shape[0])) * Xc_C.T).T)
# Dc_C = np.sum( (Dc_C * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
# Dd_C = (np.ones((Xd_C.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_C.shape[0])) * Xd_C.T).T )
# Dd_C = np.sum( (Dd_C * (self.Md.reshape(-1,1)) )**2 , axis=1 )
# D_C = (Dc_C + Dd_C).T
# MG = {}
# for i in range(Y.shape[0]):
#   #finding k closest control units to unit i
#   idx = np.argpartition(D_C[i,:],k)
# matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C = Xc_C[idx[:k],:], Xd_C[idx[:k],:], Y_C[idx[:k]], D_C[i,idx[:k]]
# #finding k closest treated units to unit i
# idx = np.argpartition(D_T[i,:],k)
# matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T = Xc_T[idx[:k],:], Xd_T[idx[:k],:], Y_T[idx[:k]],D_T[i,idx[:k]]
# MG[i] = {'unit':[ Xc[i], Xd[i], Y[i], T[i] ] ,'control':[ matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C],'treated':[matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T ]}
# return MG
#
# def CATE(self,MG,outcome_discrete=False,model='linear'):
#   cate = {}
# for k,v in MG.items():
#   #control
#   matched_X_C = np.hstack((v['control'][0],v['control'][1]))
# matched_Y_C = v['control'][2]
# #treated
# matched_X_T = np.hstack((v['treated'][0], v['treated'][1]))
# matched_Y_T = v['treated'][2]
# x = np.hstack(([v['unit'][0]], [v['unit'][1]]))
# if not outcome_discrete:
#   if model=='mean':
#   yt = np.mean(matched_Y_T)
# yc = np.mean(matched_Y_C)
# cate[k] = {'CATE': yt - yc,'outcome':v['unit'][2],'treatment':v['unit'][3] }
# if model=='linear':
#   yc = lm.Ridge().fit( X = matched_X_C, y = matched_Y_C )
# yt = lm.Ridge().fit( X = matched_X_T, y = matched_Y_T )
# cate[k] = {'CATE': yt.predict(x) - yc.predict(x),'outcome':v['unit'][2],'treatment':v['unit'][3] }
# if model=='RF':
#   yc = ensemble.RandomForestRegressor().fit( X = matched_X_C, y = matched_Y_C )
# yt = ensemble.RandomForestRegressor().fit( X = matched_X_T, y = matched_Y_T )
# cate[k] = {'CATE': yt.predict(x)[0] - yc.predict(x)[0],'outcome':v['unit'][2],'treatment':v['unit'][3] }
# return pd.DataFrame.from_dict(cate,orient='index')
