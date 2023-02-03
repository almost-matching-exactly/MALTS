#' Matching After Learning To Stretch
#'
#' Implementation of the Matching After Learning To Stretch (MALTS) algorithm.
#'
#' @param data Data to be matched. Either a data frame or a path to a .csv file
#'   to be read into a data frame. Treatment must be described by a logical or
#'   binary numeric column with name \code{treated_column_name}. If supplied,
#'   outcome must be described by a column with name \code{outcome}.
#'   The outcome will be treated as continuous if numeric with more than two
#'   values, as binary if a two-level factor or numeric with values 0 and 1
#'   exclusively, and as multi-class if a factor with more than two levels. If
#'   the outcome column is omitted, matching will be performed but treatment
#'   effect estimation will not be possible. All columns not containing outcome
#'   or treatment will be treated as covariates for matching.
#' @param outcome Name of the outcome column in \code{holdout} and
#'   also in \code{data}, if supplied in the latter. Defaults to 'outcome'.
#' @param treatment Name of the treatment column in \code{data} and
#'   \code{holdout}. Defaults to 'treated'.
#' @param discrete Specifies which of the covariates in \code{data} are
#'   discrete. If supplied, either a vector of column names or a vector of
#'   column indices corresponding to the discrete covariates. If not supplied,
#'   all covariate columns that are factors will be assumed to be discrete and
#'   the remaining will be assumed to be continuous.
#' @param C Regularization weight for the optimization; defaults to 1.
#' @param k_tr,k_est Determine matched group sizes for training, matching,
#'   respectively. \code{k_*} many control units \emph{and} \code{k_*} many
#'   treated units are included in each group.
#' @param reweight A logical scalar denoting if treated and control groups
#'   should be reweighted during training, according to their respective samples
#'   sizes. Defaults to \code{FALSE}.
#' @param n_repeats,n_folds Integers controlling how many times the MALTS
#'   algorithm should be run. The tasks of 1. distance metric learning and 2.
#'   matching are performed \code{n_folds x n_repeats} many times. In each run
#'   (corresponding to \code{n_repeats}), the data is randomly split into
#'   \code{n_folds} many folds. Each fold is used once for distance metric
#'   learning and the remainder of the time for matching.
#' @param estimate_CATEs,smooth_CATEs Logical scalars determining: 1. if CATEs
#'   should be estimated from the matched data and 2. if so, whether the CATEs
#'   should be smoothed. For very large inputs, estimating CATEs in the call to
#'   \code{MALTS} will be much faster than computing them after the fact. If no
#'   smoothing is done, estimated CATEs are averages across the CATEs yielded by
#'   different stretch matrices and the associated standard deviations are
#'   sample standard deviations. Otherwise, smoothing is done via linear mean /
#'   quantile regression. The estimated CATEs are the fitted values from the
#'   mean regression of the unsmoothed CATE estimates on the covariates. The
#'   associated standard deviation is given by 1/4 the difference between the
#'   fitted values from 5% and 95% quantile regressions of the same form.
#' @param missing_data Specifies how to handle missingness in \code{data}. If
#'   'none' (default), assumes no missing data. If 'drop', effectively drops
#'   units with missingness from the data and does not match them (they will
#'   still appear in the matched dataset that is returned, however). If
#'   'impute', imputes the missing data via \code{mice::mice}.
#' @param impute_with_treatment,impute_with_outcome Logical scalars. If
#'   \code{TRUE}, uses treatment, outcome, respectively, to impute covariates
#'   when either \code{missing_data} or \code{missing_holdout} are \code{TRUE}.
#'   Default to \code{TRUE}, \code{FALSE}, respectively.
#' @param ... A  named list of additional arguments to be passed to
#'   \code{nloptr::cobyla}. These control the details of the optimization
#'   procedure, such as a maximum number of evaluations and convergence
#'   tolerance. \code{MALTS} defaults are \code{maxeval = 500} and
#'   \code{xtol_rel = 1e-4}, with all other terms set to \code{nloptr} defaults.
#'   For more details, see \code{nloptr::nl.opts}.
#'
#' @details
#' \code{MALTS} implements the Matching After Learning To Stretch algorithm of
#' Parikh, Rudin, and Volfovsky (JMLR 2022), which solves an optimization
#' problem to learn a distance metric / stretch matrix for units that
#' prioritizes variables more predictive of the outcome and then matches
#' accordingly. Additional details can be found in the paper
#' \href{https://www.jmlr.org/papers/volume23/21-0053/21-0053.pdf}{here}. In
#' this implementation, the data is split into \code{n_folds} many folds, with
#' each fold being used once to learn the distance metric and the remainder of
#' the time to match the units therein. This entire procedure is repeated
#' \code{n_repeats} times, generating a total of \code{n_folds x n_repeats} many
#' stretch matrices and matched groups for each unit. All the conditional
#' average treatment effects (CATEs) from different stretch matrices are
#' averaged together prior to being returned and can further be
#' regression-smoothed via \code{smooth_CATEs}.
#'
#' \code{print.malts} gives information about convergence of the optimization
#' and an estimate of the average treatment effect.
#'
#' \code{plot.malts} displays two plots by default. The first shows the diagonal
#' entries of the stretch matrix \eqn{M} used to define distance between units,
#' with boxplots over multiple runs (\code{n_repeats}) and train-test splits
#' (\code{n_folds}). The second plots a density estimate of the estimated CATE
#' distribution, where the CATEs are (possibly smoothed) averages over multiple
#' runs (\code{n_repeats}) or train-test splits (\code{n_folds}).
#'
#' \code{plot_CATE} plots estimated CATEs against a feature of choice, allowing
#' for conditioning on a subset of the data.
#'
#' @name MALTS
#' @return An object of class \code{malts}, which is a list of four entries:
#' * data: The originally supplied data, with additional `CATE` and `sd_CATE`
#' columns, as per the \code{estimate_CATEs} and \code{smooth_CATEs} arguments.
#' * M: A matrix whose columns refer to different covariates and whose rows
#' correspond to their stretch values in the stretch matrix M across different
#' runs of the MALTS algorithm (as determined by \code{n_repeats} and
#' \code{n_folds}).
#' * MGs: A list of matched groups with each entry corresponding to matches
#' formed under a different stretch matrix (as determined by \code{n_repeats}
#' and \code{n_folds}). \code{MGs[[i]][[j]]} is a vector of the units matched to
#' \code{j} according the \code{i}'th stretch matrix.
#' * info: Miscellaneous information about the call to \code{MALTS} and
#' associated output.
#'
#' @examples
#' malts_out <- MALTS(gen_data(n = 500, p = 10))
#' print(malts_out)
#' plot(malts_out)
#' @importFrom stats model.matrix predict rbinom rnorm var complete.cases median
#'   density
#' @importFrom utils flush.console read.csv write.csv
#' @importFrom graphics abline axis barplot boxplot legend lines points
#' @importFrom stats as.formula loess.smooth
NULL
#> NULL
#' @rdname MALTS
#' @export
#'
MALTS <- function(data, outcome = 'outcome', treatment = 'treated',
                  discrete, C = 1, k_tr = 15, k_est = 50, reweight = FALSE,
                  n_folds = 5, n_repeats = 1, estimate_CATEs = TRUE,
                  smooth_CATEs = TRUE,
                  missing_data = c('none', 'drop', 'impute'),
                  impute_with_outcome = FALSE, impute_with_treatment = FALSE,
                  ...) {

  missing_data <- match.arg(missing_data)

  out <- preprocess(data, treatment, outcome, discrete, C, k_tr,
                    k_est, reweight, estimate_CATEs,
                    missing_data,
                    impute_with_outcome, impute_with_treatment)

  data <- out$data
  info <- out$info

  covariates <- !(colnames(data) %in% c(treatment, outcome, 'missing'))

  discrete <- info$discrete
  continuous <- setdiff(colnames(data),
                        c(discrete, treatment, outcome, 'missing'))


  malts_out <- vector('list', length = 4)
  names(malts_out) <- c('data', 'info', 'M', 'MGs')
  info$convergence <- vector('numeric', n_folds * n_repeats)
  malts_out$M <- matrix(NA, nrow = n_folds * n_repeats, ncol = sum(covariates))
  colnames(malts_out$M) <- colnames(data)[covariates]
  malts_out$MGs <- vector('list', length = n_folds * n_repeats)

  CATEs <- matrix(NA, nrow = n_folds * n_repeats, ncol = nrow(data))
  for (i in seq_len(n_repeats)) {
    folds <- stratified_k_fold(data, n_folds)
    for (j in seq_len(n_folds)) {
      ind <- j + (i - 1) * n_folds
      tmp <- MALTS.int(data[folds != j, ], data[folds == j, ],
                       outcome, treatment, discrete, continuous, info,
                       C, k_tr, k_est,
                       reweight, estimate_CATEs)

      malts_out$M[ind, ] <- tmp$M
      malts_out$MGs[[ind]] <- tmp$MGs
      CATEs[ind, folds != j] <- tmp$CATEs
      info$convergence[ind] <- tmp$convergence
    }
  }

  # Make sure discrete covariates are factors
  avg_CATE <- colMeans(CATEs, na.rm = TRUE)
  data$CATE <- avg_CATE

  form <- paste('CATE ~', paste(c(discrete, continuous), collapse = '+'))
  if (smooth_CATEs) {
    fit <- rq(form, tau = c(0.05, 0.95), data = data)
    sd_CATE <- (fit$fitted.values[, 2] - fit$fitted.values[, 1]) / 4
    stopifnot(all(sd_CATE > 0))
    avg_CATE <- lm(form, data = data)$fitted.values
  }
  else {
    sd_CATE <- apply(CATEs, 2, sd, na.rm = TRUE)
  }
  data$sd_CATE <- sd_CATE
  data$missing <- NULL
  malts_out$data <- data
  malts_out$info <- info
  class(malts_out) <- 'malts'
  return(malts_out)
}

MALTS.int <- function(data, holdout, outcome,
                      treatment, discrete, continuous, info,
                      C, k_tr, k_est, reweight, estimate_CATEs, ...) {
  treated <- holdout[treatment] == 1
  control <- holdout[treatment] == 0

  Y_T <- holdout[outcome][treated]
  Y_C <- holdout[outcome][control]

  n_T <- length(Y_T)
  n_C <- length(Y_C)

  df_T <- holdout[treated, ]
  df_C <- holdout[control, ]

  Xc_T <- df_T[continuous]
  Xc_C <- df_C[continuous]

  Xd_T <- df_T[discrete]
  Xd_C <- df_C[discrete]

  Dc_T <- make_dist_array(Xc_T, FALSE)
  Dc_C <- make_dist_array(Xc_C, FALSE)
  Dd_T <- make_dist_array(Xd_T, TRUE)
  Dd_C <- make_dist_array(Xd_C, TRUE)

  fit_out <-
    fit(outcome, treatment, holdout, k_tr, discrete, C, reweight,
        Dc_T, Dc_C, Dd_T, Dd_C, Y_T, Y_C, ...)

  Mc <- fit_out$Mc
  Md <- fit_out$Md
  M <- fit_out$M

  info$convergence <- fit_out$convergence

  n <- nrow(data)
  ##### Should reconsider making discrete correspond to the whole data ######

  treated <- data[[treatment]] == 1
  control <- data[[treatment]] == 0

  inds_t <- which(treated)
  inds_c <- which(control)

  data_T <- data[inds_t, ]
  data_C <- data[inds_c, ]

  Y_T <- data[[outcome]][treated]
  Y_C <- data[[outcome]][control]

  Xc <- data[continuous]
  Xd <- data[discrete]

  Xc_T <- data_T[continuous]
  Xc_C <- data_C[continuous]

  Xd_T <- data_T[discrete]
  Xd_C <- data_C[discrete]

  # Checked
  # This results in a *transposed* version of e.g. line 140 in pymalts.py
  # Can we really not do this in one step somehow?
  Dc_T <- rep_mat(Xc, nrow(Xc_T)) - aperm(rep_mat(Xc_T, nrow(Xc)), c(1, 3, 2))
  Dc_T <- colSums((Dc_T * Mc) ^ 2)

  Dd_T <- rep_mat(Xd, nrow(Xd_T)) != aperm(rep_mat(Xd_T, nrow(Xd)), c(1, 3, 2))
  Dd_T <- colSums((Dd_T * Md) ^ 2)
  D_T <- Dc_T + Dd_T

  Dc_C <- rep_mat(Xc, nrow(Xc_C)) - aperm(rep_mat(Xc_C, nrow(Xc)), c(1, 3, 2))
  Dc_C <- colSums((Dc_C * Mc) ^ 2)

  Dd_C <- rep_mat(Xd, nrow(Xd_C)) != aperm(rep_mat(Xd_C, nrow(Xd)), c(1, 3, 2))
  Dd_C <- colSums((Dd_C * Md) ^ 2)
  D_C <- Dc_C + Dd_C

  MGs <- vector('list', length = n)
  # browser()
  if (info$estimate_CATEs && info$outcome_type == 'continuous') {
    Tr <- data[[treatment]]
    Y <- data[[outcome]]

    CATEs <- numeric(nrow(data))
  }
  weights <- numeric(nrow(data))

  for (i in 1:n) {

    if (data$missing[i]) {
      MGs[[i]] <- NULL
      if (info$estimate_CATEs && info$outcome_type == 'continuous') {
        CATEs[i] <- NA
      }
      next
    }

    # Don't match self or missing units
    dists <- D_C[i, ]
    dists[dists == 0 | is.na(dists)] <- Inf

    MG <-  inds_c[dists <= sort(dists, partial = k_est)[k_est]]

    dists <- D_T[i, ]
    dists[dists == 0 | is.na(dists)] <- Inf
    # Partial sorting doesn't allow for index.return
    MG <- c(MG, inds_t[dists <= sort(dists, partial = k_est)[k_est]])

    MG <- c(i, MG)

    if (info$estimate_CATEs) {
      CATEs[i] <- estimate_CATEs(i, Tr, Y, MG, info)
    }

    weights[MG] <- weights[MG] + 1

    MGs[[i]] <- MG
  }

  # if (info$estimate_CATEs && info$outcome_type %in% c('binary', 'continuous')) {
  # data$CATE <- CATEs
  # }
  # data$weight <- weights

  return(list(MGs = MGs, M = M, weights = weights, CATEs = CATEs,
              convergence = fit_out$convergence))
  # malts_out <- postprocess(data, MGs, M, info)
}
