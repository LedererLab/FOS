#ifndef ISTA_H
#define ISTA_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
// Boost Headers
//
// SPAMS Headers
//
// Armadillo Headers
//
// Project Specific Headers
#include "fos_debug.h"

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template < typename T >
T pos_part( T x ) {
    return std::max( x, 0.0 );
}

template < typename T >
T soft_threshold( T x, T y ) {
    auto sgn_T = static_cast<T>( sgn(x) );
    return sgn_T*pos_part( std::abs(x) - y );
}

template < typename T >
T prox( T x, T lambda ) {
    return ( std::abs(x) >= lambda )?( x - static_cast<T>( sgn( x ) )*lambda ):( 0 );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > soft_threshold_mat( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > mat, T lambda ) {

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat_x( mat.rows(), mat.cols() );

    for( uint i = 0; i < mat.rows() ; i ++ ) {

        for( uint j = 0; j < mat.cols() ; j++ ) {
            mat_x( i, j ) =  prox( mat( i, j ), lambda );
        }
    }

    return mat_x;
}

template < typename T >
T f_beta ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X, \
           Eigen::Matrix< T, Eigen::Dynamic, 1  > Y, \
           Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta ) {

    auto arg = Y - X*Beta;
    return arg.squaredNorm();

}

template < typename T >
T f_beta_tilda ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X, \
                 Eigen::Matrix< T, Eigen::Dynamic, 1  > Y, \
                 Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta, \
                 Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_prime,
                 T L ) {

    auto f_beta = X*Beta_prime - Y;
    T taylor_term_0 = f_beta.squaredNorm();

    auto f_grad = 2*X.transpose()*( X*Beta_prime - Y );
    auto beta_diff = ( Beta - Beta_prime );

    auto taylor_term_1 = f_grad.transpose()*beta_diff;

    auto taylor_term_2 = L/2.0*beta_diff.squaredNorm();

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T >
T backtrace( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X, \
             Eigen::Matrix< T, Eigen::Dynamic, 1  > Y, \
             Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k, \
             Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k_less_1,
             T L,
             T eta ) {

    uint counter = 0;
    T new_L = L;

    DEBUG_PRINT( "Value of f: " << f_beta( X, Y, Beta_k ) << " Value of f tilde: " << f_beta_tilda( X, Y, Beta_k, Beta_k_less_1, L ) );

    while( f_beta( X, Y, Beta_k ) <= f_beta_tilda( X, Y, Beta_k, Beta_k_less_1, new_L ) ) {
        counter ++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );
        new_L = eta*new_L;
    }

    return new_L;
}

//    def ista(A, b, l, maxit):
//        Beta = np.zeros(X.shape[1])
//        L = linalg.norm(X) ** 2  # Lipschitz constant

//        for _ in Betarange(maBetait):
//            Beta = soft_thresh(Beta + np.dot(X.T, Y - X.dot(Beta)) / L, l / L)

//    return Beta

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > update_beta (
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic  > X, \
    Eigen::Matrix< T, Eigen::Dynamic, 1  > Y, \
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta, \
    T L, \
    T thres ) {

    auto X_t = X.transpose();
    auto X_beta = X*Beta;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > beta_to_modify = Beta + 1.0/L*( X_t*( Y - X_beta ) );
    return soft_threshold_mat( beta_to_modify, thres/L );

}

template < typename T >
T naive_L( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X ) {

    auto arg = X.transpose() * X;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > > eigensolver( arg );

    auto eigen_values = eigensolver.eigenvalues();

    T lambda_1 = 2.0*eigen_values.maxCoeff();

    return std::abs( lambda_1 );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X, \
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y, \
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_0, \
        uint num_iterations, \
        T L_0, \
        T lambda ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    for( uint i = 0; i < num_iterations; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k_less_1 = Beta;
        Beta = update_beta( X, Y, Beta, L, 0.5*lambda );

        while( f_beta( X, Y, Beta ) > f_beta_tilda( X, Y, Beta, Beta_k_less_1, L ) ) {
            DEBUG_PRINT( "Beta: \n"  << Beta );

            static uint counter = 0;
            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L = eta*L;
            DEBUG_PRINT( "L: " << L );

            Beta_k_less_1 = Beta;
            Beta = update_beta( X, Y, Beta, L, 0.5*lambda );

        }

    }

    return Beta;

}

#endif // ISTA_H
