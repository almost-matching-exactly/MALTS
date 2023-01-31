check_args <-
  function(data, holdout, C,
           impute_with_outcome, impute_with_treatment, info) {

    discrete <- info$discrete
    treated_column_name <- info$treatment
    outcome_column_name <- info$outcome
    replace <- FALSE

    if (!is.data.frame(data)) {
      stop('`data` must be a data frame or a character denoting a .csv file ',
           'in the working directory.')
    }
    if (!is.data.frame(holdout)) {
      stop('`holdout` must be a data frame, a character denoting a .csv ' ,
           'file in the working directory, or a numeric proportion of ',
           '`data` to use as a holdout set.')
    }

    if (info$estimate_CATEs && info$outcome_type != 'continuous') {
      warning('CATEs are not computed if the outcome is not continuous. ',
              'Set `estimate_CATEs` equal to FALSE to suppress this warning.')
    }

    if (info$outcome_type != 'none') {
      outcome <- data[[outcome_column_name]]
      if (is.numeric(outcome) && length(unique(outcome)) == 2) {
        if (!all(sort(unique(outcome)) == c(0, 1))) {
          stop('If the outcome column is binary and numeric, it must be coded ',
               'with values 0 and 1.', call. = FALSE)
        }
      }
    }

    if (length(unique(data[[treated_column_name]])) == 1) {
      stop('`data` must contain both treated and control units.')
    }

    if (length(unique(holdout[[treated_column_name]])) == 1) {
      stop('`holdout` must contain both treated and control units.')
    }

    data_cols <- colnames(data)
    holdout_cols <- colnames(holdout)

    # Seems like I can remove but maybe there's a reason I put it up here...?
    if (!(outcome_column_name %in% holdout_cols)) {
      stop('`holdout` must contain outcome column with name ',
           '`outcome_column_name`')
    }

    cov_inds_data <- which(!(colnames(data) %in%
                               c(treated_column_name, outcome_column_name)))
    cov_inds_holdout <- which(!(colnames(holdout) %in%
                                  c(treated_column_name, outcome_column_name)))

    if (any(!(discrete %in% data_cols))) {
      stop('Supplied a variable name in `discrete` that is not present in ',
           '`data`.', call. = FALSE)
    }

    if (!identical(colnames(data)[cov_inds_data],
                   colnames(holdout[cov_inds_holdout]))) {
      stop('Covariate columns of `data` and `holdout` ',
           'must have have the same names and be in the same order.')
    }

    if (!is.numeric(C) | C < 0 | is.infinite(C)) {
      stop('C must be a finite, nonnegative scalar.')
    }

    if (!is.character(treated_column_name)) {
      stop('`treated_column_name` must be a character.')
    }

    if (!(treated_column_name %in% data_cols)) {
      stop('`treated_column_name` must be the name of a column in `data.`')
    }

    if (!(treated_column_name %in% holdout_cols)) {
      stop('`treated_column_name` must be the name of a column in `holdout.`')
    }

    if (is.factor(data[[treated_column_name]])) {
      stop('Treated variable in `data` must be numeric binary or logical.')
    }

    if (is.factor(holdout[[treated_column_name]])) {
      stop('Treated variable in `holdout` must be numeric binary or logical.')
    }

    if (!is.character(outcome_column_name)) {
      stop('`outcome_column_name` must be a character.')
    }

    if (info$outcome_type != 'none' & !(outcome_column_name %in% data_cols)) {
      stop('`outcome_column_name` must be the name of a column in `data.` ',
           'if outcome is supplied.')
    }

    if (!(outcome_column_name %in% holdout_cols)) {
      stop('`outcome_column_name` must be the name of a column in `holdout.`')
    }

    if (!is.logical(replace)) {
      stop('`replace` must be a logical scalar')
    }

    # if (!(verbose %in% c(0, 1, 2, 3))) {
    #   stop('`verbose` must be one of: 0, 1, 2, 3')
    # }

    ## Missing data parameters
    if (info$missing_data == 'impute' &&
        !requireNamespace("mice", quietly = TRUE)) {
      stop("Package `mice` needed to impute missing values. Please install it",
           " or select different options for `missing_data`.",
           call. = FALSE)
    }
    else if (info$missing_holdout == 'impute' &&
             !requireNamespace("mice", quietly = TRUE)) {
      stop("Package `mice` needed to impute missing values. Please install it",
           " or select different options for `missing_holdout`.",
           call. = FALSE)
    }

    if (impute_with_outcome & info$outcome_type == 'none') {
      stop('Outcome not present in `data`; ',
           'cannot request to use it to impute missingness.')
    }

    if (!is.logical(impute_with_outcome)) {
      stop('`impute_with_outcome` must be a logical scalar')
    }

    if (!is.logical(impute_with_treatment)) {
      stop('`impute_with_outcome` must be a logical scalar')
    }
  }
