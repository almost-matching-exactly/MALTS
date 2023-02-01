read_data <- function(data, treated_column_name, outcome_column_name) {
  if (is.character(data)) {
    tryCatch(
      error = function(cnd) {
        stop('Cannot read `data` .csv file from working directory')
      },
      data <- read.csv(data, header = TRUE)
    )
  }
  return(data)
}
