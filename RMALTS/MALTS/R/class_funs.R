BLUE <- '#56B4E9'
ORANGE <- '#E69F00'

convert_nlopt_code <- function(val) {
  if (val == 1) {
    'Success'
  }
  else if (val == 2) {
    'Stopval Reached'
  }
  else if (val == 3) {
    'Function Tolerance Reached'
  }
  else if (val == 4) {
    'X Tolerance Reached'
  }
  else if (val == 5) {
    'Max Evaluations Reached'
  }
  else if (val == 6) {
    'Max Time Reached'
  }
  else if (val == -1) {
    'Failure'
  }
  else if (val == -2) {
    'Invalid Arguments'
  }
  else if (val == -3) {
    'Out of Memory'
  }
  else if (val == -4) {
    'Roundoff Errors'
  }
  else if (val == -5) {
    'Forced Stop'
  }
  else {
    'Unrecognized Stopping Criterion'
  }
}

#' Matched Groups from MALTS
#'
#' Create, Print, and Plot Matched Groups from a MALTS fit.
#' @param unit Query unit whose matched group is desired
#' @param MALTS_out An object of type \code{malts}
#' @param threshold_n,threshold_p How often -- across multiple runs and multiple
#'   train-test splits -- a unit must have been matched to another in order to
#'   be considered within the latter's matched group. \code{threshold_p}
#'   specifies a proportion and \code{threshold_n} specifies a number of times.
#'   Defaults to \code{threshold_n = 1}.
#' @details
#'
#' \code{print.mg.malts} is equivalent to \code{print.summary.mg.malts}, which
#' displays information about 1. the size of the query unit's matched group and
#' 2. pruned and unpruned CATE estimates. Unpruned estimates use all the units
#' the query unit was ever matched to, while pruned estimates only use the units
#' satisfying the specified thresholds (with equivalence if \code{threshold_n =
#' 1} or \code{threshold_p = 0}).
#'
#' \code{plot.mg.malts} plots information about the joint distribution of two
#' covariates within a matched group.
#'
#' @name MGs
#' @return \code{make_MG} returns an object of type \code{mg.malts}, which is a
#' list including the entries:
#' * data: treatment, outcome, covariate, and CATE information for the units in
#' the matched group
#' * query: the unit whose matched group is returned
#' * threshold: the minimum number of matches required for inclusion in the
#' matched group. This is the same as \code{threshold_n}, if provided;
#' otherwise, \code{threshold_p} is converted accordingly.
#'
#' All other entries are as for objects of type \code{malts}
#'
#' @examples
#' malts_out <- MALTS(gen_data(n = 500, p = 10), n_folds = 5, n_repeats = 3)
#' # only including units matched at least 50% of the time
#' mg <- make_MG(1, malts_out, threshold_p = 0.5)
#' print(mg)
#' plot(mg, 1, 2)
#'
NULL
#> NULL
#' @rdname MGs
#' @export
make_MG <- function(unit, MALTS_out, threshold_n, threshold_p) {

  if (!missing(threshold_n) & !missing(threshold_p)) {
    stop('Only one of `threshold_n` and `threshold_p` can be supplied.')
  }
  if (!missing(threshold_p)) {
    # ignoring folds where you were only used for estimation
    times_matched <- sum(sapply(MALTS_out$MGs, function(x) !is.null(x[[1]])))
    threshold <- threshold_p * times_matched
  }
  else {
    threshold <- ifelse(missing(threshold_n), 1, threshold_n)
  }

  # All MGs of that unit
  MGs <- lapply(MALTS_out$MGs, function(x) x[[unit]])

  n_times_matched <- table(unlist(MGs))
  above_threshold <- as.integer(names(which(n_times_matched >= threshold)))
  MG <- MALTS_out$data[above_threshold, ]

  info <- MALTS_out$info
  info$algo <- NULL
  info$missing_data <- NULL
  info$missing_holdout <- NULL
  info$replacement <- NULL

  out <- list(data = MG, info = info, M = MALTS_out$M,
              query = unit, threshold = ceiling(threshold))
  class(out) <- 'mg.malts'
  return(out)
}

#' Print a Matched Group
#'
#' @param x An object of class \code{mg.malts}.
#' @param digits Number of significant digits for printing the CATE estimate.
#' @param linewidth Maximum number of characters on line; output will be wrapped
#' accordingly.
#'
#' @export
#' @rdname MGs
print.mg.malts <-
  function(x, digits = getOption('digits'), linewidth = 80, ...) {
  print(summary.mg.malts(x), digits = digits, linewidth = linewidth, ...)
}

#' Summarize a Matched Group
#'
#' @param object An object of class \code{mg.malts}
#' @param ... Additional arguments to be passed on to other methods. Not used.
#'
#' @rdname MGs
#' @export
summary.mg.malts <- function(object, ...) {

  out <- list()
  info <- object$info

  ####### Think about what makes the most sense for M. Average? Min/max over all
  ####### runs? even those not involving a match with a given unit on that run?
  # covs <- names(object$M)
  #
  # # Fix data.matrix once we've figured out what to do with factors here
  # X <- data.matrix(object$data[covs])
  #
  # disc <- info$discrete
  # cont <- setdiff(covs, disc)
  #
  # # Make sure the first is always the MG
  # diffs_cont <- sweep(X[, cont], 2, X[, cont][1, ])
  # diffs_disc <- sweep(X[, disc], 2, X[, disc][1, ], function(x, y) x != y) * 1
  # diffs <- matrix(nrow = nrow(X), ncol = length(covs))
  # colnames(diffs) <- covs
  # for (j in seq_along(covs)) {
  #   if (covs[j] %in% disc) {
  #     diffs[, j] <- diffs_disc[, covs[j]]
  #   }
  #   else {
  #     diffs[, j] <- diffs_cont[, covs[j]]
  #   }
  # }
  #
  # # Memory efficiency...
  # distances <- diag(diffs %*% diag(object$M) %*% t(diffs))
  # distances <- distances[distances > 0] # eliminate self
  #
  # out$distance <- list('min' = min(distances),
  #                      'median' = median(distances),
  #                      'max' = max(distances))

  # Or store in info?
  # out$query <- rownames(object$data)[1]
  Tr <- object$data[[object$info$treatment]]

  if (object$info$outcome_type == 'continuous') {
    Y <- object$data[[object$info$outcome]]
  }
  Tr <- object$data[[object$info$treatment]]

  n_match_t <- sum(Tr == 1) - (Tr[1] == 1)
  n_match_c <- sum(Tr == 0) - (Tr[1] == 0)

  out$n_matches <- c(n_match_c, n_match_t)

  if (object$info$outcome_type == 'continuous') {
    out$CATE_pruned <- mean(Y[Tr == 1]) - mean(Y[Tr == 0])
    out$CATE_unpruned <- object$data$CATE[object$query]
  }
  out$threshold <- object$threshold
  out$query <- object$query
  class(out) <- 'summary.mg.malts'
  return(out)
}

#' Print a Matched Group Summary
#'
#' @param x An object of class \code{summary.mg.malts}, generated by a call to
#' the function of the same name.
#' @param digits Number of significant digits for printing.
#' @param linewidth Maximum number of characters on line; output will be wrapped
#' accordingly.
#' @rdname MGs
#' @export
print.summary.mg.malts <-
  function(x, digits = getOption('digits'), linewidth = 80, ...) {
    indentation <- 2
    cat(paste0('The main matched group of unit ', x$query, ':\n'))
    cat(strwrap(paste(' ', 'Matched to', sum(x$n_matches), 'units, ',
                      x$n_matches[2], ' treated and ',
                      x$n_matches[1], ' control.'),
                width = linewidth, indent = indentation, exdent = indentation),
        sep = '\n ')

    if ('CATE_pruned' %in% names(x)) {
      cat(strwrap(paste0(' ', 'The unpruned CATE estimate is: ',
                         round(x$CATE_unpruned, digits = digits), '.'),
                  width = linewidth,
                  indent = indentation, exdent = indentation),
          sep = '\n ')
      cat(strwrap(paste0(' ', 'The pruned CATE estimate (with a threshold of ',
                         x$threshold, ') is: ',
                         round(x$CATE_pruned, digits = digits), '.'),
                  width = linewidth,
                  indent = indentation, exdent = indentation),
          sep = '\n ')
    }

    return(invisible(x))

    lablen <- 12
    cat('\nMatched Group Diameters:\n')
    cat(format('  Minimum', width = lablen),
        format(x$distance$min, digits = digits),
        '\n')
    cat(format('  Median', width = lablen),
        format(x$distance$median, digits = digits),
        '\n')
    cat(format('  Maximum', width = lablen),
        format(x$distance$max, digits = digits),
        '\n')

    return(invisible(x))
}

get_average_effects <- function(x) {

  n <- length(x$MGs)
  Tr <- x$data[[x$info$treatment]]
  Y <- x$data[[x$info$outcome]]

  K <- numeric(n)

  for (i in 1:n) {
    Tr_i <- Tr[i]

    for (j in 1:n) {
      if (Tr[j] == Tr_i) {
        next
      }
      MG <- x$MGs[[j]]
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

  for (i in 1:n) {
    cond_var_tmp <- 0
    cond_var_tmp_t <- 0
    cond_var_tmp_c <- 0

    Tr_i <- Tr[i]
    Y_i <- Y[i]

    MG <- x$MGs[[i]]
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

#' @param x An object of class \code{malts}, returned by a call to
#'   \code{\link{MALTS}}.
#' @param digits Number of significant digits for printing the average treatment
#' effect.
#' @param linewidth Maximum number of characters on line; output will be wrapped
#' accordingly.
#' @param ... Additional arguments to be passed to other methods.
#' @rdname MALTS
#' @export
print.malts <- function(x, digits = getOption('digits'), linewidth = 80, ...) {

  df <- x$data

  algo <- x$info$algo

  outcome_type <- x$info$outcome_type

  replacement <- x$info$replacement

  # n_iters <- length(x$cov_sets)

  # n_matched <- sum(df$matched)
  n_matched <- sum(sapply(x$MGs, function(x) length(x) > 0))
  n_total <- nrow(df)

  indentation <- 2
  cat('An object of class `malts`:\n')
  # cat(strwrap(paste(' ', 'MALTS matched ',
  #                   n_matched, 'out of', n_total, 'units',
  #                   ifelse(replacement, 'with', 'without'), 'replacement.'),
  #             width = linewidth, indent = indentation, exdent = indentation),
  #     sep = '\n ')

  #######
  if (all(x$info$convergence > 0)) {
    success_status <- 'successfully'
  }
  else if (all(x$info$convergence < 0)) {
    success_status <- 'unsuccessfully'
  }
  else {
    success_status <- 'with_mixed_success'
  }
  cat(paste('  MALTS terminated', success_status), '\n')

  n_convergence_types <- table(x$info$convergence)
  cat('  Optimization stopped', '\n')
  for (i in seq_along(n_convergence_types)) {
    times <- n_convergence_types[i]
    cat(paste('   ', times, ifelse(times == 1, 'time', 'times'), 'due to:',
              convert_nlopt_code(as.integer(names(times)))),
        '\n')
  }

  if (outcome_type == 'continuous' & x$info$estimate_CATEs) {

    cat(strwrap(paste0('  The average treatment effect of `',
                       x$info$treatment, '` on `', x$info$outcome,
                       '` is estimated to be ',
                       round(mean(x$data$CATE, na.rm = TRUE),
                             digits = digits),
                       '.'),
                width = linewidth, indent = indentation, exdent = indentation),
        sep = '\n ')
  }
  # if (outcome_type == 'continuous') {
  #   cat(strwrap(paste0('  The average treatment effect of `',
  #                      x$info$treatment, '` on `', x$info$outcome,
  #                      '` is estimated to be ',
  #                      round(ifelse(x$info$estimate_CATEs &&
  #                                     outcome_type == 'continuous',
  #                                   mean(x$data$CATE, na.rm = TRUE),
  #                                   get_average_effects(x)['All', 'Mean']),
  #                            digits = digits),
  #                      '.'),
  #               width = linewidth, indent = indentation, exdent = indentation),
  #       sep = '\n ')
  # }

  if (x$info$missing_data == 'drop') {
    missing_data_message <-
      'Units with missingness in the matching data were not matched.'
  }
  else if (x$info$missing_data == 'impute') {
    missing_data_message <-
      'Missing values in the matching data were imputed by MICE.'
  }
  else if (x$info$missing_data == 'none') {
    missing_data_message <- NULL
  }

  if (!is.null(missing_data_message)) {
    cat(strwrap(missing_data_message,
                width = linewidth, indent = indentation, exdent = indentation),
        sep = '\n ')
  }

  return(invisible(x))
}

# MG Diameters: min, max, median / mean
  # diameter is max distance between query unit and other units in MG

#' @param x An object of class \code{malts}, returned by a call to
#'   \code{\link{MALTS}}.
#' @param which_plots A vector describing which plots should be displayed. See
#' details.
#' @param ... Additional arguments to passed on to other methods.
#' @rdname MALTS
#' @export
plot.malts <- function(x, which_plots = c(1, 2), ...) {

  if (min(which_plots) <= 0 || max(which_plots) >= 3) {
    stop('Please supply an integer 1 through 2 for `which_plots`.')
  }

  n_plots <- length(which_plots)
  n_plotted <- 0
  first_plot <- min(which_plots)

  MGs <- x$MGs
  n_MGs <- length(MGs)
  Tr <- x$data[[x$info$treatment]]
  Y <- x$data[[x$info$outcome]]

  # Really don't need this if they're not asking for 2
  if (!x$info$estimate_CATEs) {
    CATEs <- numeric(n_MGs)
    for (i in seq_len(nrow(x$data))) {
      MG <- MGs[[i]]
      if (is.null(MG)) {
        CATEs[i] <- NA
        next
      }
      if (Tr[i] == 1) {
        CATEs[i] <- Y[i] - mean(Y[MG[Tr[MG] == 0]])
      }
      else {
        CATEs[i] <- mean(Y[MG[Tr[MG] == 1]] - Y[i])
      }
    }
  }
  else {
    CATEs <- x$data$CATE
  }
  ATE <- mean(CATEs)

  if (1 %in% which_plots) {
    boxplot(x$M, xlab = 'Feature', ylab = 'Stretch Factor', col = BLUE)
    # barplot(x$M, xlab = 'Feature', ylab = 'Stretch Factor')
    n_plotted <- n_plotted + 1
  }

  if (n_plotted == n_plots) {
    return(invisible(x))
  }

  if (2 %in% which_plots) {
    if (interactive() & first_plot != 2) {
      readline(prompt = "Press <enter> to view next plot")
    }
    dens <- density(CATEs, na.rm = TRUE)
    if (0 > min(dens$x) && 0 < max(dens$x)) {
      include_null <- TRUE
    }
    else {
      include_null <- FALSE
    }

    plot(dens,
         xlab = c('Estimated Conditional Average Treatment Effect'),
         ylab = '', main = '',
         zero.line = FALSE)

    abline(v = ATE, lty = 2, col = BLUE)
    if (include_null) {
      abline(v = 0, lty = 3, col = ORANGE)
    }

    if (include_null) {
      legend('topright',
             legend = c('Estimated ATE', 'Null Effect'),
             lty = c(2, 3), col = c(BLUE, ORANGE))
    }
    else {
      legend('topright', legend = c('Estimated ATE'), lty = 2, col = BLUE)
    }
  }
  return(invisible(x))
}


#' Plot a Matched Group
#'
#' @param x An object of type \code{mg.malts}, returned by a call to \code{MG}.
#' @param cov1,cov2 Covariates, specified either by indices or column names in
#'   \code{x$data} to plot against.
#' @param smooth A logical scalar denoting whether smoothed loess estimates
#'   should be added to the plots. Defaults to \code{FALSE}.
#' @param ... Additional arguments to be passed on to other methods.
#' @importFrom graphics image
#' @rdname MGs
#' @export
plot.mg.malts <- function(x, cov1, cov2, smooth = FALSE, ...) {

  stopifnot(typeof(cov1) == typeof(cov2)) # ?

  if (is.numeric(cov1)) {
    cov1 <- colnames(x$data)[cov1]
    cov2 <- colnames(x$data)[cov2]
  }

  discrete <- x$info$discrete
  data <- x$data
  if (xor(cov1 %in% discrete, cov2 %in% discrete)) {
    bad <- c() # underrepresented categories
    if (cov1 %in% discrete) {
      form <- as.formula(paste(cov2, '~', x$info$treatment, '+', cov1))
      xlab <- cov1
      disc_vals <- unique(data[, cov1])
    }
    else {
      form <- as.formula(paste(cov1, '~', x$info$treatment, '+', cov2))
      xlab <- cov2
      disc_vals <- unique(data[, cov2])
    }
    for (val in disc_vals) {
      if (sum(data[, xlab] == val) == 1) { # xlab is the discrete cov
        bad <- c(bad, val)
      }
    }
    # not sure this can happen...
    if (length(bad) == length(disc_vals)) {
      stop(paste('Not enough variation in', xlab, '. Choose a new covariate.'))
    }

    data <- data[!(data[, xlab] %in% bad), ]
    data[, xlab] <- droplevels(data[, xlab])

    boxplot(form, data = data, xaxs = FALSE, col = c(BLUE, ORANGE),
            xaxt = 'n', xlab = xlab)
    axis(1, at = seq(1.5, 2 * length(disc_vals) - 0.5, by = 2),
         labels = disc_vals)
    legend('topright', legend = c('control', 'treated'), pch = c(16, 16),
           col = c(BLUE, ORANGE))
  }
  else if (cov1 %in% discrete && cov2 %in% discrete) {
    unique1 <- unique(x$data[, cov1])
    unique2 <- unique(x$data[, cov2])
    n1 <- length(unique1)
    n2 <- length(unique2)

    mean_mat <- matrix(nrow = n1, ncol = n2)
    for (i in seq_len(n1)) {
      for (j in seq_len(n2)) {
        inds <- x$data[, cov1] == unique1[i] & x$data[, cov2] == unique2[j]
        if (sum(inds) == 0) {
          mean_mat[i, j] <- NA
          next
        }
        tmp <- x$data[inds, ]
        mean_mat[i, j] <- mean(tmp[[x$info$outcome]])
      }
    }
    # browser()
    # fields::image.plot(mean_mat, col = viridisLite::viridis(16))
    image(mean_mat)
  }
  else {
    treated <- x$data[[x$info$treatment]] == 1

    # par(mar = c(5, 4, 6, 2) + 0.1, xpd = TRUE)
    plot(x$data[treated, cov1], x$data[treated, cov2],
         col = ORANGE, xlab = cov1, ylab = cov2)
    points(x$data[!treated, cov1], x$data[!treated, cov2], col = BLUE)

    if (smooth) {
      loess_out <- loess.smooth(x$data[treated, cov1], x$data[treated, cov2])
      lines(loess_out$x, loess_out$y, col = ORANGE, lwd = 1.5)
      loess_out <- loess.smooth(x$data[!treated, cov1], x$data[!treated, cov2])
      lines(loess_out$x, loess_out$y, col = BLUE, lwd = 1.5)
    }

    legend('topright', legend = c('control', 'treated'), pch = c(1, 1),
           col = c(BLUE, ORANGE))
  }
}

#' @param x An object of class \code{malts}, returned by a call to
#'   \code{MALTS}.
#' @param cov1 The variable that CATEs should be plotted against. Either an
#'   index of \code{x$data} corresponding to a covariate or an
#'   appropriate column name.
#' @param condition_on A named vector describing what variables to condition on
#'   when plotting CATEs. Names correspond to the variables and how they should
#'   be filtered and values correspond to what values they should be filtered
#'   with respect to. For example c("X1<=" = 5, "X2==" = 0) only plots CATEs for
#'   those units satisfying X1 <= 5 and X2 == 0. Accepted operators are "<",
#'   "<=", ">", ">=", "==", and "!=",  as well as "=" and "", which are
#'   interpreted as "==". See examples.
#' @param df A data frame from which CATEs and the corresponding covariate
#'   values are taken to be plotted instead of \code{x$data}. Useful if
#'   more complex filtering than permitted by \code{condition_on} is desired.
#' @param smooth A logical scalar, denoting whether a LOESS curve should be fit
#'   to the covariate-CATE data if \code{cov1} is continuous. Defaults to FALSE.
#' @rdname MALTS
#' @export
plot_CATE <- function(x, cov1, condition_on, df, smooth = FALSE) {
  if (!missing(condition_on) & !missing(df)) {
    stop('Only one of `condition_on` or `df` can be specified')
  }

  # && instead of & to short circuit
  if ((!missing(df) && !('CATE' %in% colnames(df))) |
      (missing(df) && !('CATE' %in% colnames(x$data)))) {
    stop("CATEs have not been computed")
  }

  # support for < and > not just <= >=
  # should smoothed estimates be computed based off the whole data or no?
  if (!missing(condition_on)) {
    df <- x$data
    conditioning_vars <- names(condition_on)
    for (i in seq_along(condition_on)) {
      v <- conditioning_vars[i]
      last <- substr(v, nchar(v), nchar(v))
      if (last == "<") {
        compare <- `<`
        end_at <- nchar(v) - 1
      }
      else if (last == ">") {
        compare <- `>`
        end_at <- nchar(v) - 1
      }
      else if (last == "=") {
        end_at <- nchar(v) - 2
        penult <- substr(v, nchar(v) - 1, nchar(v) - 1)
        if (penult == "<") {
          compare <- `<=`
        }
        else if (penult == ">") {
          compare <- `>=`
        }
        else if (penult == "=") {
          compare <- `==`
        }
        else if (penult == "!") {
          compare <- `!=`
        }
        else {
          compare <- `==`
          end_at <- nchar(v) - 1
        }
      }
      v <- substr(v, 1, end_at)
      df <- df[compare(df[, v], condition_on[i]), ]
    }
  }
  if (missing(df)) {
    df <- x$data
  }

  if (is.numeric(cov1)) {
    cov1 <- colnames(df)[cov1]
  }

  if (cov1 %in% x$info$discrete) {
    boxplot(as.formula(paste('CATE ~', cov1)),
            data = df, col = c(BLUE),
            xlab = cov1)
  }
  else {
    plot(df[, cov1], df$CATE, col = BLUE,
         ylab = 'CATE', xlab = cov1)
    if (smooth) {
      loess_out <- loess.smooth(df[, cov1], df$CATE)
      lines(loess_out$x, loess_out$y, lwd = 1.5, col = ORANGE)
    }
  }
}

# plot_CATE <- function(CATE, cov, smooth = FALSE) {
#   plot(CATE, cov)
#   if (smooth) {
#     loess_out <- loess.smooth(cov, CATE)
#     lines(loess_out$x, loess_out$y, lwd = 1.5)
#   }
# }

#' Summarize the output of MALTS
#'
#' These methods create and print objects of class \code{summary.malts}
#' containing information on the matched groups formed by the MALTS algorithm,
#' and, if applicable, average treatment effects.
#'
#' The average treatment effect (ATE) is estimated as the average CATE estimate
#' across all matched units in the data, while the average treatment effect on
#' the treated (ATT) and average treatment effect on controls (ATC) average only
#' across matched treated or matched control units, respectively. Variances of
#' these estimates are computed as in Abadie, Drukker, Herr, and Imbens (The
#' Stata Journal, 2004) assuming constant treatment effect and homoscedasticity.
#' Note that the implemented estimator is \strong{not} =asymptotically normal
#' and so in particular, asymptotically valid confidence intervals or hypothesis
#' tests cannot be conducted on its basis. In the future, the estimation
#' procedure will be changed to employ the nonparametric regression bias
#' adjustment estimator of Abadie and Imbens 2011 which is asymptotically
#' normal.
#'
#' @return A list of type \code{summary.malts} with the following entries:
#' \describe{
#' \item{TEs}{
#'   If the matching data had a continuous outcome, estimates of the ATE, ATT,
#'   and ATC and the corresponding variance of the estimates.
#'  }
#' }
#' @importFrom stats setNames
#' @name summary.malts
NULL
#> NULL
#'
#' @param object An object of class \code{malts}, returned by a call to
#'   \code{\link{MALTS}}.
#' @param ... Additional arguments to be passed on to other methods
#' @rdname summary.malts
#' @export
summary.malts <- function(object, ...) {
  summary_obj <- list()
  if (object$info$outcome_type == 'continuous') {
    # average_effects <- get_average_effects(object)
    # summary_obj$TEs <- average_effects
    summary_obj$ATE <- mean(object$data$CATE)
  }

  # average stretches
  M <- colMeans(object$M)

  stretch_d <- M[object$info$discrete]
  stretch_c <- M[!(names(M) %in% object$info$discrete)]

  stretches <- list()
  if (length(object$info$discrete) > 0) {
    stretches <-
      c(stretches,
              list('discrete' = setNames(range(stretch_d),
                                    c(names(stretch_d)[which.min(stretch_d)],
                                      names(stretch_d)[which.max(stretch_d)]))))
  }
  if (length(object$info$discrete) < length(M)) {
    stretches <-
      c(stretches,
        list('continuous' = setNames(range(stretch_c),
                                      c(names(stretch_c)[which.min(stretch_c)],
                                      names(stretch_c)[which.max(stretch_c)]))))
  }

  summary_obj$stretch <- stretches
  # summary_obj$stretch <-
  #   list('discrete' = setNames(range(stretch_d),
  #                              c(names(stretch_d)[which.min(stretch_d)],
  #                                names(stretch_d)[which.max(stretch_d)])),
  #        'continuous' = setNames(range(stretch_c),
  #                                c(names(stretch_c)[which.min(stretch_c)],
  #                                  names(stretch_c)[which.max(stretch_c)])))
  summary_obj$info <- object$info
  class(summary_obj) <- 'summary.malts'
  return(summary_obj)
}

#' Print a summary of MALTS
#'
#' @param x An object of class \code{summary.malts}, returned by a call to
#'   \code{\link{summary.malts}}
#' @param digits Number of significant digits for printing the average treatment
#' effect estimates and their variances.
#' @param ... Additional arguments to be passed on to other methods.
#' @rdname summary.malts
#' @export
print.summary.malts <- function(x, digits = 3, ...) {
  max_meanlen <- 7
  max_varlen <- 8
  lablen <- 13

  if ('ATE' %in% names(x)) {
    cat(strwrap(paste0('  The average treatment effect of `',
                       x$info$treatment, '` on `', x$info$outcome,
                       '` is estimated to be ',
                       round(mean(x$ATE, na.rm = TRUE), digits = digits),
                       '.'),
                width = 80, indent = 2, exdent = 2),
        sep = '\n ')
  }

  if ('TEs' %in% names(x)) {

    ATE_meanstr <- format(x$TEs['All', 1], digits = digits, justify = 'right')
    ATE_varstr <- format(x$TEs['All', 2], digits = digits, justify = 'right')

    ATT_meanstr <- format(x$TEs['Treated', 1], digits = digits, justify='right')
    ATT_varstr <- format(x$TEs['Treated', 2], digits = digits, justify='right')
    ATC_meanstr <- format(x$TEs['Control', 1], digits = digits, justify='right')
    ATC_varstr <- format(x$TEs['Control', 2], digits = digits, justify='right')

    max_meanlen <- max(max_meanlen,
                       nchar(c(ATE_meanstr, ATT_meanstr, ATC_meanstr)))
    max_varlen <- max(max_varlen, nchar(c(ATE_varstr, ATT_varstr, ATC_varstr)))

    if ('TEs' %in% names(x)) {

      cat('\nAverage Treatment Effects:\n')
      cat(format('', width = lablen),
          format('Mean', width = max_meanlen, justify = 'right'),
          format('Variance', width = max_varlen, justify = 'right'),
          '\n')

      cat(format('  All', width = lablen),
          format(x$TEs['All', 1], digits = digits,
                 width = max_meanlen, justify = 'right'),
          format(x$TEs['All', 2], digits = digits,
                 width = max_varlen, justify = 'right'),
          '\n')

      cat(format('  Treated', width = lablen),
          format(x$TEs['Treated', 1],
                 digits = digits, width = max_meanlen, justify = 'right'),
          format(x$TEs['Treated', 2],
                 digits = digits, width = max_varlen, justify = 'right'),
          '\n')

      cat(format('  Control', width = lablen),
          format(x$TEs['Control', 1],
                 digits = digits, width = max_meanlen, justify = 'right'),
          format(x$TEs['Control', 2],
                 digits = digits, width = max_varlen, justify = 'right'),
          '\n')
    }
  }

  if ('discrete' %in% names(x$stretch)) {
    stretch_d <- x$stretch$discrete
  }
  if ('continuous' %in% names(x$stretch)) {
    stretch_c <- x$stretch$continuous
  }

  cat('\nAverage Stretch Values:\n')

  min_d_strlen <- -1
  min_c_strlen <- -1
  pad <- 2
  if ('discrete' %in% names(x$stretch)) {
    min_d_str <- paste0(format(stretch_d[1], digits = digits),
                               ' (', names(stretch_d)[1], ') ')
    min_d_strlen <- nchar(min_d_str)

    max_d_str <- paste0(format(stretch_d[2], digits = digits),
                               ' (', names(stretch_d)[2], ') ')

    n_pre_Minimum <- pad + nchar('Discrete ')
  }
  if ('continuous' %in% names(x$stretch)) {
    min_c_str <- paste0(format(stretch_c[1], digits = digits),
                        ' (', names(stretch_c)[1], ') ')
    min_c_strlen <- nchar(min_c_str)

    max_c_str <- paste0(format(stretch_c[2], digits = digits),
                               ' (', names(stretch_c)[2], ') ')

    n_pre_Minimum <- pad + nchar('Continuous ')
  }

  # chars from start of line to "Maximum"
  n_pre_Maximum <- n_pre_Minimum + max(min_c_strlen, min_d_strlen)
  # chars between end of "Minimum", start of "Maximum
  n_Min_Max <- max(min_c_strlen, min_d_strlen)

  cat(paste0(rep(' ', n_pre_Minimum), collapse = ''),
      'Minimum',
      paste0(rep(' ', n_pre_Maximum - n_pre_Minimum - nchar('Minimum')), collapse = ''),
      'Maximum',
      '\n', sep = '')

  if ('continuous' %in% names(x$stretch)) {
    cat(strrep(' ', pad), 'Continuous ',
        min_c_str,
        strrep(' ', n_Min_Max - min_c_strlen),
        max_c_str, '\n',
        sep = '')
  }

  if ('discrete' %in% names(x$stretch)) {
    cat(strrep(' ', pad), 'Discrete',
        strrep(' ', n_pre_Minimum - nchar('Discrete') - pad),
        min_d_str,
        strrep(' ', n_Min_Max - min_d_strlen),
        max_d_str, '\n',
        sep = '')
  }
}

