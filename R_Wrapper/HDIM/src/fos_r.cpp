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
#include "Solvers/SubGradientDescent/ISTA/ista.hpp"
#include "Solvers/SubGradientDescent/FISTA/fista.hpp"
#include "Solvers/CoordinateDescent/coordinate_descent.hpp"

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > NumVect2Eigen( const Rcpp::NumericVector& vec ) {

    // int len = vec.length();
    //
    // Eigen::Matrix< T, Eigen::Dynamic, 1 > matrixOutput(len);
    //
    // //#pragma omp parallel for
    // for( int j = 0; j < len; j++ ) {
    //     matrixOutput( j ) = static_cast< T >( vec( j ) );
    // }
    //
    // return matrixOutput;

    int len = vec.length();

    T* non_const_vec_data = new T[ len ];
    const T* vec_data = &vec[0];

    std::copy( vec_data, vec_data + len, non_const_vec_data );

    return Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 1 > >( non_const_vec_data, len );
}

template < typename T >
Rcpp::NumericVector Eigen2NumVec( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& vec ) {

    // int rows = vec.rows();
    //
    // Rcpp::NumericVector vectorOutput(rows);
    //
    // //#pragma omp parallel for
    // for( int i = 0; i < rows; i++ ) {
    //     vectorOutput( i ) = static_cast<double>( vec( i ) );
    // }
    //
    // return vectorOutput;

    const T* vect_data = vec.data();
    return Rcpp::NumericVector( vect_data, vect_data + vec.size() );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > NumMat2Eigen( const Rcpp::NumericMatrix& mat ) {
    //
    // int rows = mat.rows();
    // int cols = mat.cols();
    //
    // Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > matrixOutput(rows,cols);
    //
    // //#pragma omp parallel for collapse(2)
    // for( int i = 0; i < rows; i++ ) {
    //     for( int j = 0; j < cols; j++ ) {
    //         matrixOutput( i,j ) = static_cast< T >( mat( i,j ) );
    //     }
    // }
    //
    // return matrixOutput;
    //
    int rows = mat.rows();
    int cols = mat.cols();

    int eigen_mat_size = rows*cols;

    T* non_const_mat_data = new T[ eigen_mat_size ];
    const T* mat_data = &mat[0];

    std::copy( mat_data, mat_data + eigen_mat_size, non_const_mat_data );

    return Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >( non_const_mat_data, rows, cols );

}

template < typename T >
Rcpp::NumericMatrix Eigen2NumMat( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat ) {

    // int rows = mat.rows();
    // int cols = mat.cols();
    //
    // Rcpp::NumericMatrix matrixOutput(rows,cols);
    //
    // //#pragma omp parallel for collapse(2)
    // for( int i=0; i<rows; i++ ) {
    //     for( int j = 0; j < cols; j++ ) {
    //         matrixOutput( i,j ) = static_cast<double>( mat( i,j ) );
    //     }
    // }
    //
    // return matrixOutput;

    const T* mat_data = mat.data();
    return Rcpp::NumericMatrix( mat_data, mat_data + mat_data.size() );
}

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List FOS( const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& Y, const std::string solver_type ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);

    hdim::experimental::X_FOS<double> fos;

    if ( solver_type == "ista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::ista );
    } else if ( solver_type == "fista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::fista );
    } else if ( solver_type == "cd" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::cd );
    } else if ( solver_type == "lazy_cd" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::lazy_cd );
    } else {
      fos(  mat_X, vect_Y, hdim::SolverType::ista );
    }

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

// [[Rcpp::export]]
Rcpp::NumericVector CoordinateDescent( Rcpp::NumericMatrix X,
                        Rcpp::NumericVector Y,
                        Rcpp::NumericVector Beta_0,
                        double lambda,
                        unsigned int num_iterations ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::CoordinateDescentSolver<double> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector CoordinateDescent_DG( Rcpp::NumericMatrix X,
                        Rcpp::NumericVector Y,
                        Rcpp::NumericVector Beta_0,
                        double lambda,
                        double duality_gap_target ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::CoordinateDescentSolver<double> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, static_cast<double>( duality_gap_target ) ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector ISTA( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          unsigned int num_iterations,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector ISTA_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, static_cast<double>( duality_gap_target ) ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTA( Rcpp::NumericMatrix X,
                           Rcpp::NumericVector Y,
                           Rcpp::NumericVector Beta_0,
                           double lambda,
                           unsigned int num_iterations,
                           double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double> fista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTA_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::FISTA<double> fista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, static_cast<double>( duality_gap_target ) ) );
}

#endif // FOS_R_H
