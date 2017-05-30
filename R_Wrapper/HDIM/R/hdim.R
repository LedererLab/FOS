AppFOS <- function( X, Y ) {

	fos_results <- HDIM::FOS( as.matrix(X), as.matrix(Y) )

	# betas <- as.data.frame( fos_results$beta )
	# rownames( betas ) <- colnames( X )
	#
	betas <- c( fos_results$intercept )
	names( betas ) = c("(Intercept)")

	betas <- c( betas, fos_results$beta )
	beta_df <- data.frame(name = names(betas), betas)

	return( beta_df )

}
