postprocess <- function(data, MGs, M, info) {
  data[['missing']] <- NULL

  ret_list <- list(data = data, MGs = MGs, M = M, info = info)

  class(ret_list) <- 'malts'
  return(ret_list)
}
