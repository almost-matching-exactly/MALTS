#' Matching After Learning To Stretch
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
#' @param holdout Holdout data to be used to estimate a distance matrix. If a
#'   numeric scalar between 0 and 1, that proportion of \code{data} will be made
#'   into a holdout set and only the \emph{remaining proportion} of \code{data}
#'   will be matched. Otherwise, a data frame or a path to a .csv file. The
#'   holdout data must contain an outcome column with name
#'   \code{outcome}; other restrictions on column types are as for
#'   \code{data}. Covariate columns must have the same column names and order as
#'   \code{data}. This data will \emph{not} be matched. Defaults to 0.1.
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
#' @param estimate_CATEs A logical scalar. If \code{TRUE}, CATEs for each unit
#'   are estimated throughout the matching procedure, which will be much faster
#'   than computing them after a call to \code{MALTS} for very large inputs.
#'   Defaults to \code{TRUE}.
#' @param missing_data Specifies how to handle missingness in \code{data}. If
#'   'none' (default), assumes no missing data. If 'drop', effectively drops
#'   units with missingness from the data and does not match them (they will
#'   still appear in the matched dataset that is returned, however). If
#'   'impute', imputes the missing data via \code{mice::mice}.
#' @param missing_holdout Specifies how to handle missingness in \code{holdout}.
#'   If 'none' (default), assumes no missing data; if 'drop', drops units with
#'   missingness and does not use them to learn a distance matrix; and if
#'   'impute', imputes the missing data via \code{mice::mice}.
#' @param impute_with_treatment,impute_with_outcome A logical scalar. If
#'   \code{TRUE}, uses treatment, outcome, respectively, to impute covariates
#'   when either \code{missing_data} or \code{missing_holdout} are \code{TRUE}.
#'   Default to \code{TRUE}, \code{FALSE}, respectively. \code{missing_holdout =
#'   2}. Defaults to \code{TRUE}.
#' @param ... A  named list of additional arguments to be passed to
#'   \code{nloptr::cobyla}. These control the details of the optimization
#'   procedure, such as a maximum number of evaluations and convergence
#'   tolerance. MALTS defaults are \code{maxeval = 500} and
#'   \code{xtol_rel = 1e-4}, with all other terms set to \code{nloptr} defaults.
#'   For more details, see \code{nloptr::nl.opts}.
#'
#' @name MALTS
#' @return An object of class \code{malts}. The \code{weight} column refers to
#'   the number of matched groups units are included in.
#'
#' @examples
#' malts_out <- MALTS(gen_data(n = 500, p = 10))
#' @importFrom stats model.matrix predict rbinom rnorm var complete.cases median
#'   density
#' @importFrom utils flush.console read.csv write.csv
#' @importFrom graphics abline axis barplot boxplot legend lines points
#' @importFrom stats as.formula loess.smooth
NULL
#> NULL
#' @rdname MALTS
#' @export
MALTS <- function(data, holdout = 0.1,
                  outcome = 'outcome', treatment = 'treated',
                  discrete, C = 1, k_tr = 15, k_est = 50, reweight = FALSE,
                  estimate_CATEs = TRUE,
                  missing_data = c('none', 'drop', 'impute'),
                  missing_holdout = c('none', 'drop', 'impute'),
                  impute_with_outcome = FALSE,
                  impute_with_treatment = FALSE,
                  ...) {

  missing_data <- match.arg(missing_data)
  missing_holdout <- match.arg(missing_holdout)

  out <- preprocess(data, holdout, treatment, outcome, discrete, C, k_tr,
                    k_est, reweight, estimate_CATEs,
                    missing_data, missing_holdout,
                    impute_with_outcome, impute_with_treatment)

  data <- out$data
  holdout <- out$holdout
  info <- out$info

  covariates <- !(colnames(data) %in% c(treatment, outcome))

  discrete <- info$discrete
  continuous <- setdiff(colnames(data),
                        c(discrete, treatment, outcome, 'missing'))

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

  # units <- as.numeric(rownames(data))
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
    # browser()
    if (info$estimate_CATEs) {
      CATEs[i] <- estimate_CATEs(i, Tr, Y, MG, info)
    }

    weights[MG] <- weights[MG] + 1

    MGs[[i]] <- MG
  }

  if (info$estimate_CATEs && info$outcome_type %in% c('binary', 'continuous')) {
    data$CATE <- CATEs
  }
  data$weight <- weights

  malts_out <- postprocess(data, MGs, M, info)
  return(malts_out)
}
