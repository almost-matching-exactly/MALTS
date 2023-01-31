#' Generate Toy Data for Matching
#'
#' \code{gen_data} generates toy data that can be used to explore MALTS
#' functionality.
#'
#' \code{gen_data} simulates data in the format accepted by \code{\link{MALTS}}
#' where \eqn{\epsilon \sim N(0, I_n)}.
#'
#' @param n Number of units desired in the data set. Defaults to 250.
#' @param p Number of covariates in the data set. Must be greater than 2.
#'   Defaults to 5.
#' @param write A logical scalar. If \code{TRUE}, the resulting data is stored
#'   as a .csv file as specified by arguments \code{path} and \code{filename}.
#'   Defaults to \code{FALSE}.
#' @param path The path to the location where the data should be written if
#'   \code{write = TRUE}. Defaults to \code{getwd()}.
#' @param filename The name of the file to which the data should be written if
#'   \code{write = TRUE}. Defaults to AME.csv.
#'
#' @return A data frame that may be passed to \code{\link{MALTS}}.
#' Covariates are numeric, treatment is binary numeric and
#'   outcome is numeric.
#' @export
gen_data <- function(n = 250, p = 5,
                     write = FALSE, path = getwd(), filename = 'AME.csv') {
  if (p <= 2) {
    stop('`p` must be greater than 2')
  }
  TE <- 5

  discrete_inds <- sample(p, round(p / 2))
  continuous_inds <- setdiff(seq_len(p), discrete_inds)

  discrete_covs <-
    sample(1:4, size = n * length(discrete_inds), replace = TRUE,
           prob = c(0.2, 0.3, 0.4, 0.1))

  discrete_covs <- matrix(discrete_covs, nrow = n)

  continuous_covs <- matrix(rnorm(n * length(continuous_inds), 2, 2), nrow = n)

  # covs <- cbind(discrete_covs, continuous_covs)
  covs <- matrix(nrow = n, ncol = p)
  covs[, discrete_inds] <- discrete_covs
  covs[, continuous_inds] <- continuous_covs

  treated <- rbinom(n, 1, prob = 0.5)

  outcome <-
    15 * covs[, 1] - 10 * covs[, 2] + 5 * covs[, 3] +
    TE * treated +
    rnorm(n)

  data <- data.frame(covs, outcome = outcome, treated = treated)
  data[, discrete_inds] <- lapply(data[, discrete_inds], as.factor)

  if (write) {
    write.csv(data, file = paste0(path, '/', filename),
              row.names = FALSE)
  }
  return(data)
}
