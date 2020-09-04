library(nloptr)

d <- function(X_i, X_j, M) {
 A <- (X_i-X_j)*M
 A <- t(A) %*% A
# print(A)
 return(A[1,1])
}

knn <- function(M, K, X, Tr, X_i, T_i){
  D_i <- matrix(Inf, nrow(X),1)
  for (j in 1:nrow(X)) {
    if(Tr[j]==T_i) {
      D_i[j,1] <- d(X_i,X[j,], M)
    }
  }
  D_K <- sort(D_i)[K]

  NN_i <- as.integer(D_i <= D_K)
  return(NN_i)
}

delta <- function(M,outcome,treatment,data,K){
  Y <- data[, outcome]
  Tr <- data[, treatment]
  X <- data.matrix(data[, !(names(data) %in% c(outcome, treatment))])
  l <- matrix(1,nrow(X),1)
  for(i in 1:nrow(X)){
    NN_i <- knn(M,K,X,Tr,X[i,],Tr[i])
    Y_i_hat <- (t(NN_i)%*%Y)/K
    l[i,1] <- (Y[i] - Y_i_hat)^2
  }
#  print(l)
  return(sum(l))
}

#only for continuous variables
 objective <- function(M, outcome, treatment, data, K) {
  c <-0
  cons1 <- 1e+25 * sum(as.integer(M < 0))
  loss <- c*norm(M,"2") + delta(M,outcome,treatment,data,K) + cons1
  #print(M)
  return(loss)
 }

hin <- function(M){
  h <- min(M)
  return(h)
}

fit<- function(outcome, treatment, data, K) {
  p <- ncol(data)-2
  M_init <- matrix(1,1,p)
  res <- cobyla(M_init, objective, control = list(maxeval = 500), outcome=outcome, treatment=treatment, data=data, K=K)
  M_opt <- res$par
  return(M_opt)
}

M <- fit("outcome","treated", Example_training, 10)
