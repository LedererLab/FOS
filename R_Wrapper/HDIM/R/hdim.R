AppFOS <- function( X, Y ) {

	fos_results <- HDIM::FOS( as.matrix(X), as.matrix(Y) )
	support_indices <- which( fos_results$support > 0, arr.ind=TRUE )

	betas <- c( fos_results$intercept )
	names( betas ) = c("(Intercept)")

	# betas <- c( betas, fos_results$beta )
	betas <- c( betas, fos_results$beta[support_indices] )
	beta_df <- data.frame( betas, colnames = names(betas) )

	return( beta_df )

}
