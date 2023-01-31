stratified_k_fold <- function(data, k = 5) {
  inds <- numeric(nrow(data))
  treated <- data[[treated]] == 1
  n_t <- sum(treated)
  n_c <- nrow(data) - n_t
  n_t_per_fold <- ceiling(n_t / k)
  inds[treated] <- sample(rep(1:k, n_t_per_fold), n_t, FALSE)
  inds[!treated] <- sample(rep(1:k, n_c_per_fold), n_c, FALSE)
  return(inds)
}

estimate_CATEs <- function(i, Tr, Y, MG, info) {
  T_MG <- MG[Tr[MG] == 1]
  C_MG <- MG[Tr[MG] == 0]

  if (info$outcome_type == 'continuous') {
    if (Tr[i] == 1) {
      return(Y[i] - mean(Y[C_MG]))
    }
    else {
      return(mean(Y[T_MG]) - Y[i])
    }
  }
  else if (info$outcome_type == 'binary') {
    odds_t <- mean(Y[T_MG]) / mean(1 - Y[T_MG])
    odds_c <- mean(Y[C_MG]) / mean(1 - Y[C_MG])
    return(odds_t / odds_c)
  }
}

rep_mat <- function(X, n_times) {
  ### Implicitly checked via make_dist_array
  tX <- t(X)
  # array() faster than outer()
  return(array(tX, dim = c(dim(tX), n_times)))
}

make_dist_array <- function(X, discrete) {
  ### CHECKED ###
  D <- rep_mat(X, nrow(X))
  if (discrete) {
    return(D != aperm(D, c(1, 3, 2)))
  }
  return(D - aperm(D, c(1, 3, 2)))
}

threshold <- function(x, K) {
  ### CHECKED ###
  out <-
    apply(x, 2, function(z) {
      z <= sort(z, partial = K + 1)[K + 1]
    })
  return(t(out))
}

calcW <- function(Dc, Dd, Mc, Md, K) {
  ### CHECKED ###
  Dc <- colSums((Dc * Mc) ^ 2)
  Dd <- colSums((Dd * Md) ^ 2)

  W <- threshold(Dc + Dd, K)
  W <- W / (rowSums(W) - diag(W))
  return(W)
}

calc_delta <- function(Mc, Md, treatment, outcome, df, K, discrete, continuous,
                       reweight, Dc_T, Dc_C, Dd_T, Dd_C, Y_T, Y_C) {

  ### CHECKED ###

  W_T <- calcW(Dc_T, Dd_T, Mc, Md, K)
  W_C <- calcW(Dc_C, Dd_C, Mc, Md, K)

  delta_T <- sum((Y_T - (W_T %*% Y_T - diag(W_T) * Y_T)) ^ 2)
  delta_C <- sum((Y_C - (W_C %*% Y_C - diag(W_C) * Y_C)) ^ 2)

  if (reweight) {
    n_T <- length(Y_T)
    n_C <- length(Y_C)
    n <- n_T + n_C
    return(n * (delta_T / n_T + delta_C / n_C))
  }
  return(delta_T + delta_C)
}

#only for continuous variables
objective <- function(M, outcome, treatment, data, K, discrete,
                      continuous, reweight, C,
                      Dc_T, Dc_C, Dd_T, Dd_C, Y_T, Y_C) {

  ### CHECKED ###

  Mc <- M[continuous]
  Md <- M[discrete]

  p <- length(Mc) + length(Md)
  delta <- calc_delta(Mc, Md, treatment, outcome, data, K, discrete, continuous,
                      reweight, Dc_T, Dc_C, Dd_T, Dd_C, Y_T, Y_C)
  reg <- C * (sum(Mc ^ 2) + sum(Md ^ 2))
  cons <- 0 * (sum(Mc) + sum(Md) - p) ^ 2
  return(delta + reg + cons)
}

fit <- function(outcome, treatment, data, K, discrete, C, reweight,
                Dc_T, Dc_C, Dd_T, Dd_C, Y_T, Y_C, ...) {
  p <- dim(Dc_T)[1] + dim(Dd_T)[1]
  # p <- ncol(data) - 2
  M_init <- rep.int(1, p)

  treatment_ind <- which(colnames(data) == treatment)
  outcome_ind <- which(colnames(data) == outcome)

  discrete <- (colnames(data))[-c(treatment_ind, outcome_ind)] %in% discrete
  continuous <- !discrete

  # Default parameters
  # Python uses xtol_abs which doesn't exist in the R implementation
  control <- list(maxeval = 500, xtol_rel = 1e-4)
  control_names <- names(control)

  # Sub user specifications, if any
  user_control <- list(...)
  user_control_names <- names(user_control)

  for (i in seq_along(control)) {
    if (!(control_names[i] %in% user_control_names)) {
      user_control <- c(user_control, control[i])
    }
  }

  res <-
    nloptr::cobyla(M_init, objective,
                   lower = rep.int(0, p), upper = rep.int(Inf, p),
                   outcome = outcome, treatment = treatment, data = data,
                   K = K, discrete = discrete, continuous = continuous,
                   reweight = reweight, C = C,
                   Dc_T = Dc_T, Dc_C = Dc_C, Dd_T = Dd_T, Dd_C = Dd_C,
                   Y_T = Y_T, Y_C = Y_C,
                   control = user_control)

  M <- res$par
  names(M) <- setdiff(colnames(data), c(treatment, outcome))
  Mc <- M[continuous]
  Md <- M[discrete]
  return(list(Mc = Mc, Md = Md, M = M, convergence = res$convergence))
}

