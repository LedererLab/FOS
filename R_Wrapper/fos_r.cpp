// C System-Headers
//
// C++ System headers
//
// Eigen Headers
//
// Boost Headers
//
// FISTA Headers
//
// Rcpp Headers
//
// Project Specific Headers
//#include "fos_r.h"

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]

//Rcpp::NumericVector FOS( Rcpp::NumericMatrix X, Rcpp::NumericVector Y ) {

//    const Eigen::Map< Eigen::MatrixXd >  x ( Rcpp::as< Eigen::Map< Eigen::MatrixXd > >( X ) );
//    const Eigen::Map< Eigen::VectorXd >  y ( Rcpp::as< Eigen::Map< Eigen::VectorXd > >( Y ) );

//    FOS< double > algo_fos ( x, y );
//    algo_fos.Algorithm();

//    Eigen::VectorXd beta_tilde_r_tilde = algo_fos.ReturnCoefficients();

//    return Rcpp::wrap( beta_tilde_r_tilde );

//}
