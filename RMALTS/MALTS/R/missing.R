impute_missing <- function(data, outcome_in_data,
                           treated_column_name, outcome_column_name,
                           impute_with_treatment, impute_with_outcome) {

  treatment_ind <- which(colnames(data) == treated_column_name)
  outcome_ind <- which(colnames(data) == outcome_column_name)

  pred_mat <- matrix(1, nrow = ncol(data), ncol = ncol(data))
  diag(pred_mat) <- 0

  if (!impute_with_treatment) {
    pred_mat[, treatment_ind] <- 0
  }
  if (!impute_with_outcome) {
    pred_mat[, outcome_ind] <- 0
  }

  pred_mat[c(treatment_ind, outcome_ind), ] <- 0

  mice_out <-
    mice::mice(data, m = 1,
               predictorMatrix = pred_mat, printFlag = FALSE)

  imputed_data <- mice::complete(mice_out, action = 'all')[[1]]
  return(imputed_data)
}

handle_missing_data <-
  function(data, treated_column_name, outcome_column_name,
           missing_data, impute_with_treatment, impute_with_outcome) {

    outcome_in_data <- !is.null(data[[outcome_column_name]])

    # Corresponds to data only
    cov_inds <- which(!(colnames(data) %in%
                          c(treated_column_name, outcome_column_name)))

    if (outcome_in_data) {
      to_drop_data <- is.na(data[[outcome_column_name]]) |
        is.na(data[[treated_column_name]])
    }
    else {
      to_drop_data <- is.na(data[[treated_column_name]])
    }

    if (any(to_drop_data)) {
      message('Found missingness in `data` in treatment and/or outcome; ',
              'corresponding rows will be dropped.')
    }

    if (all(to_drop_data)) {
      stop('Dropping all rows in `data` due to missingness ',
           'in treatment and/or outcome.')
    }

    data <- data[!to_drop_data, ]

    orig_missing <- which(is.na(data), arr.ind = TRUE)

    if (missing_data == 'none') {
      is_missing <- FALSE
      if (sum(is.na(data)) > 0) {
        stop('Found missingness in `data` but was told to assume there ',
             'was none. Please either change `missing_data` or ',
             'supply `data` without missingness.')
      }
    }
    else if (missing_data == 'drop') {
      is_missing <- apply(data, 1, function(row) any(is.na(row)))
      if (all(is_missing)) {
        stop('All rows in `data` contain missingness. ',
             'In this case, matches may only be made if `missing_data` ',
             " = 'keep' or `missing_data` = 'impute'.")
      }
    }
    else if (missing_data == 'impute') {
      is_missing <- FALSE
      if (sum(is.na(data)) > 0) {
        message('Starting imputation of `data`\r', appendLF = FALSE)
        data <- impute_missing(data, outcome_in_data,
                               treated_column_name, outcome_column_name,
                               impute_with_treatment, impute_with_outcome)
        message('Finished imputation of `data`\r', appendLF = FALSE)
      }
      else {
        message('No missing data found; skipping imputation.')
      }
    }

    return(list(data = data,
                is_missing = is_missing,
                orig_missing = orig_missing))
  }
