n <- 10000
n_MG <- 100

Tr <- sample(c(0, 1), n, T)
pool <- 1:n
pool_t <- which(Tr == 1)
pool_c <- which(Tr == 0)

MGs <- vector('list', length = n)
for (i in seq_len(n_MG)) {

  mg <- c(sample(pool_t, round(runif(1, 1 / n, 0.001) * n)),
          sample(pool_c, round(runif(1, 1 / n, 0.001) * n)))

  MGs[mg] <- list(mg)
  pool_t <- setdiff(pool_t, mg)
  pool_c <- setdiff(pool_c, mg)
}

unmatched <- sapply(MGs, is.null)
Y <- rnorm(n)

get_average_effects <- function(unmatched, MGs, Tr, Y) {
  new_ind <- seq_along(unmatched) - cumsum(unmatched)

  Y <- Y[!unmatched]
  Tr <- Tr[!unmatched]
  MGs <- MGs[!unmatched]
  MGs <- lapply(MGs, function(z) new_ind[z])

  n <- length(MGs)

  K <- numeric(n)

  for (i in 1:n) {
    Tr_i <- Tr[i]

    for (j in 1:n) {
      if (Tr[j] == Tr_i) {
        next
      }
      MG <- MGs[[j]]
      if (!(i %in% MG)) {
        next
      }
      opp_sign <- sum(Tr[MG] == Tr_i)
      K[i] <- K[i] + 1 / opp_sign
    }
  }

  n1 <- sum(Tr == 1)
  n0 <- sum(Tr == 0)

  ATE <- sum((2 * Tr - 1) * (1 + K) * Y) / n
  ATT <- sum((Tr - (1 - Tr) * K) * Y) / n1
  ATC <- sum((Tr * K - (1 - Tr)) * Y) / n0

  cond_var <- 0
  cond_var_t <- 0
  cond_var_c <- 0

  # browser()
  for (i in 1:n) {
    cond_var_tmp <- 0
    cond_var_tmp_t <- 0
    cond_var_tmp_c <- 0

    Tr_i <- Tr[i]
    Y_i <- Y[i]

    MG <- MGs[[i]]
    MG <- MG[Tr[MG] != Tr_i]

    if (Tr_i == 1) {
      cond_var <- cond_var + mean((Y_i - Y[MG] - ATE) ^ 2)
      cond_var_t <- cond_var_t + mean((Y_i - Y[MG] - ATT) ^ 2)
    }
    else {
      cond_var <- cond_var + mean((Y[MG] - Y_i - ATE) ^ 2)
      cond_var_c <- cond_var_c + mean((Y[MG] - Y_i - ATT) ^ 2)
    }
  }

  cond_var <- cond_var / (2 * n)
  cond_var_t <- cond_var_t / (2 * n1)
  cond_var_c <- cond_var_c / (2 * n0)

  V_sample_sate <- sum(cond_var * (1 + K) ^ 2) / n ^ 2
  V_sample_satt <- sum(cond_var_t * (Tr - (1 - Tr) * K) ^ 2) / n1 ^ 2
  V_sample_satc <- sum(cond_var_c * (Tr * K - (1 - Tr)) ^ 2) / n0 ^ 2

  return(matrix(c(ATE, ATT, ATC, V_sample_sate, V_sample_satt, V_sample_satc),
                ncol = 2,
                dimnames = list(c('All', 'Treated', 'Control'),
                                c('Mean', 'Variance'))))
}
