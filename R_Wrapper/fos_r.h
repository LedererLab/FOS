#ifndef FOS_R_H
#define FOS_R_H

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
#include <Rcpp.h>
// Project Specific Headers
#include "../FOS/fos_imperative.h"

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]

Rcpp::NumericVector FOS( Rcpp::NumericMatrix X, Rcpp::NumericVector Y ) {

    Eigen::Map<Eigen::MatrixXd> mat_X = Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X);
    Eigen::Map<Eigen::VectorXd> vect_Y = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(Y);

    Eigen::MatrixXd betas = hdim::experimental::FOS( mat_X, vect_Y );

    Rcpp::NumericVector Betas(Rcpp::wrap(betas));

    return Betas;

}

#endif // FOS_R_H
