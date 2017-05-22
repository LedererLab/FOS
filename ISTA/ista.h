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

namespace hdim {

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T >
T f_beta (
    const MatrixT<T>& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const VectorT<T>& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T >
T f_beta_tilda (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    const VectorT<T>& Beta_prime,
    T L ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_beta = X*Beta_prime - Y;
    T taylor_term_0 = f_beta.squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_grad = 2.0*X.transpose()*( f_beta );
    Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta - Beta_prime );

    T taylor_term_1 = f_grad.transpose()*beta_diff;

    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T >
VectorT<T> update_beta_ista (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T thres ) {

    VectorT<T> f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    VectorT<T> beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

template < typename T >
MatrixT<T> ISTA (
    const MatrixT<T>& X, \
    const VectorT<T>& Y, \
    const VectorT<T>& Beta_0, \
    uint num_iterations, \
    T L_0, \
    T lambda ) {

    static_assert(std::is_floating_point< T >::value, "ISTA can only be used with floating point types.");

    T eta = 1.5;
    T L = L_0;

    VectorT<T> Beta = Beta_0;

    for( uint i = 0; i < num_iterations; i++ ) {

        uint counter = 0;

        VectorT<T> Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( ( f_beta( X, Y, Beta_temp ) > f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_ista( X, Y, Beta, L, lambda );

    }

    return Beta;

}

namespace optimize {

template < typename T >
MatrixT<T> ISTA (
    const MatrixT<T>& X, \
    const VectorT<T>& Y, \
    const VectorT<T>& Beta_0, \
    uint num_iterations, \
    T L_0, \
    T lambda ) {

    static_assert(std::is_floating_point< T >::value, "ISTA can only be used with floating point types.");

    T eta = 1.5;
    T L = L_0;

    VectorT<T> Beta = Beta_0;

    for( uint i = 0; i < num_iterations; i++ ) {

        VectorT<T> f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
        VectorT<T> Beta_temp = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

        T f_beta = ( X*Beta_temp - Y ).squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > f_part = X*Beta - Y;
        T taylor_term_0 = f_part.squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta_temp - Beta );

        T taylor_term_1 = f_grad.transpose()*beta_diff;

        T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

        T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        while( f_beta > f_beta_tilde ) {

            L*= eta;

            Beta_temp = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

            f_beta = ( X*Beta_temp - Y ).squaredNorm();;

            beta_diff = ( Beta_temp - Beta );
            taylor_term_1 = f_grad.transpose()*beta_diff;
            taylor_term_2 = L/2.0*beta_diff.squaredNorm();

            f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        }

        Beta = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

    }

    return Beta;
}

}

}


#endif // ISTA_H
