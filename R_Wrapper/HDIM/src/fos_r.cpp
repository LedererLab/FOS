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
// Rcpp Headers
#include <Rcpp.h>
// Project Specific Headers
#include "FOS/x_fos.hpp"

 template < typename T >
 Eigen::Matrix< T, Eigen::Dynamic, 1 > NumVect2Eigen( const Rcpp::NumericVector& vec ) {

     int len = vec.length();

     Eigen::Matrix< T, Eigen::Dynamic, 1 > matrixOutput(len);

    //#pragma omp parallel for
     for( int j = 0; j < len; j++ ) {
         matrixOutput( j ) = static_cast< T >( vec( j ) );
     }

     return matrixOutput;
 }

template < typename T >
Rcpp::NumericVector Eigen2NumVec( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& vec ) {

    int rows = vec.rows();

    Rcpp::NumericVector vectorOutput(rows);

    //#pragma omp parallel for
    for( int i = 0; i < rows; i++ ) {
        vectorOutput( i ) = static_cast<double>( vec( i ) );
    }

    return vectorOutput;
}

 template < typename T >
 Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > NumMat2Eigen( const Rcpp::NumericMatrix& vec ) {

     int rows = vec.rows();
     int cols = vec.cols();

     Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > matrixOutput(rows,cols);

    //#pragma omp parallel for collapse(2)
     for( int i = 0; i < rows; i++ ) {
         for( int j = 0; j < cols; j++ ) {
             matrixOutput( i,j ) = static_cast< T >( vec( i,j ) );
         }
     }

     return matrixOutput;
 }

template < typename T >
Rcpp::NumericMatrix Eigen2NumMat( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat ) {

    int rows = mat.rows();
    int cols = mat.cols();

    Rcpp::NumericMatrix matrixOutput(rows,cols);

    //#pragma omp parallel for collapse(2)
    for( int i=0; i<rows; i++ ) {
        for( int j = 0; j < cols; j++ ) {
            matrixOutput( i,j ) = static_cast<double>( mat( i,j ) );
        }
    }

    return matrixOutput;
}

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]

Rcpp::List FOS( Rcpp::NumericMatrix X, Rcpp::NumericVector Y ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);

    hdim::experimental::X_FOS<double> fos;
    fos(  mat_X, vect_Y  );

    Rcpp::NumericVector beta = Eigen2NumVec<double>( fos.ReturnCoefficients() );
    beta.attr("names") = Rcpp::colnames(X);

    unsigned int stopping_index = fos.ReturnOptimIndex();
    double lambda = fos.ReturnLambda();
    double intercept = fos.ReturnIntercept();
    Rcpp::NumericVector support = Eigen2NumVec<int>( fos.ReturnSupport() );

    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("index") = stopping_index,
                              Rcpp::Named("lambda") = lambda,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("support") = support);

}

#endif // FOS_R_H
