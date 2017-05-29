AppFOS <- function( X, Y ) {

	fos_results <- HDIM::FOS( as.matrix(X), as.matrix(Y) )

	betas <- as.data.frame( fos_results$beta )
	rownames( betas ) <- colnames( X )
	
	beta_tail <- c( fos_results$intercept )
	names( beta_tail ) = c("(Intercept)")

	append( betas,beta_tail )
	
	return( betas )

}
