preprocess <- function(data, holdout, treated_column_name, outcome_column_name,
                       discrete, C, k_tr, k_est, reweight, estimate_CATEs,
                       missing_data, missing_holdout,
                       impute_with_outcome, impute_with_treatment) {
  # Get matching and holdout data, from input .csv files, if necessary
  read_data_out <-
    read_data(data, holdout, treated_column_name, outcome_column_name)

  data <- read_data_out[[1]]
  holdout <- read_data_out[[2]]

  if (is.null(outcome_column_name) || is.null(data[[outcome_column_name]])) {
    outcome_type <- 'none'
  } else if (length(unique(data[[outcome_column_name]])) == 2) {
    outcome_type <- 'binary'
  } else if (is.factor(data[[outcome_column_name]])) {
    outcome_type <- 'categorical'
  } else {
    outcome_type <- 'continuous'
  }

  if (missing(discrete)) {
    cov_inds_data <- which(!(colnames(data) %in%
                               c(treated_column_name, outcome_column_name)))

    cov_names <- colnames(data)[cov_inds_data]
    discrete <- cov_names[vapply(data[, cov_inds_data], is.factor, logical(1))]
  }
  else if (is.numeric(discrete)) {
    discrete <- colnames(data)[discrete]
  }
  else {
    if (!is.character(discrete)) {
      stop('`discrete`, if supplied, must be a character or numeric vector.',
           call. = FALSE)
    }
  }

  info <- list('algo' = 'MALTS',
               'treatment' = treated_column_name,
               'outcome' = outcome_column_name,
               'reweight' = reweight, ## Add replacement
               'estimate_CATEs' = estimate_CATEs,
               'missing_data' = missing_data,
               'missing_holdout' = missing_holdout,
               'outcome_type' = outcome_type,
               'discrete' = discrete)

  # Make sure the user didn't do anything funny
  check_args(data, holdout, C,
             impute_with_outcome, impute_with_treatment, info)

  # Impute missing data, if requested, else, prepare to deal with missingness
  #   as specified by missing_data
  missing_out <-
    handle_missing_data(data, holdout,
                        treated_column_name, outcome_column_name,
                        missing_data, missing_holdout,
                        impute_with_treatment, impute_with_outcome)

  data <- missing_out[[1]]
  holdout <- missing_out[[2]]
  is_missing <- missing_out[[3]]
  orig_missing <- missing_out[[4]]

  # orig_missing[, 'col'] <- match(orig_missing[, 'col'], ord)

  # n <- nrow(data)
  # data$matched <- rep(FALSE, n)
  # data$weight <- rep(0, n)
  data$missing <- is_missing
  # tmp_df$MG <- rep(0, n)
  # data$CATE <- rep(NA, n)

  return(list(data = data, holdout = holdout, info = info))
}
