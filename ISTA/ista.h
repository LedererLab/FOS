#ifndef ISTA_H
#define ISTA_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
// Boost Headers
//
// SPAMS Headers
//
// OpenMP Headers
#include <omp.h> //OpenMP pragmas
// Project Specific Headers
#include "../Generic/debug.h"
#include "../Generic/generics.h"

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
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > soft_threshold_mat( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat, T lambda ) {

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic > mat_x( mat.rows(), mat.cols() );

//    #pragma omp parallel for collapse(2)
    for( uint i = 0; i < mat.rows() ; i ++ ) {

        for( uint j = 0; j < mat.cols() ; j++ ) {
            mat_x( i, j ) =  prox( mat( i, j ), lambda );
        }
    }

    return mat_x;
}

template < typename T >
T f_beta ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
           const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y, \
           const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > arg = Y - X*Beta;
    return arg.squaredNorm();

}

template < typename T >
T f_beta_tilda ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                 const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y, \
                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
                 T L ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_beta = X*Beta_prime - Y;
    T taylor_term_0 = f_beta.squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_grad = 2.0*X.transpose()*( X*Beta_prime - Y );
    Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta - Beta_prime );

    T taylor_term_1 = f_grad.transpose()*beta_diff;

    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
    T L, \
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( Y - X*Beta ) );

    return soft_threshold_mat<T>( Beta + (1.0/L)*f_grad, thres/L );

//    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta + (1.0/L)*f_grad;

//    return soft_threshold_mat<T>( beta_to_modify, thres/L );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, \
        uint num_iterations, \
        T L_0, \
        T lambda ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    for( uint i = 0; i < num_iterations; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k_less_1 = Beta;

        uint counter = 0;

//        while( true ) {

//            counter++;

//            DEBUG_PRINT( "Backtrace iteration: " << counter );

//            Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

//            if( f_beta( X, Y, Beta ) > f_beta_tilda( X, Y, Beta, Beta_k_less_1, L ) ) {
//                L = L*eta;
//                DEBUG_PRINT( "L: " << L );
//            } else {
//                break;
//            }
//        }

        Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( f_beta( X, Y, Beta ) > f_beta_tilda( X, Y, Beta, Beta_k_less_1, L ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        }
    }

    return Beta;

}


#endif // ISTA_H
