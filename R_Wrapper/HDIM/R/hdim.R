AppFOS <- function( X, Y ) {

	fos_results <- HDIM::FOS( as.matrix(X), as.matrix(Y) )
	beta <- as.data.frame( fos_results$beta )
	rownames( beta ) <- colnames( X )
	return( beta )

}
