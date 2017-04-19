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

template < typename T >
T f_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T >
T f_beta_tilda (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
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
Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_ista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, \
    uint num_iterations, \
    T L_0, \
    T lambda ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

//    #pragma omp parallel for
    for( uint i = 0; i < num_iterations; i++ ) {

        uint counter = 0;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        counter++;
        //DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( ( f_beta( X, Y, Beta_temp ) > f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

            counter++;
            //DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_ista( X, Y, Beta, L, lambda );

    }

    return Beta;

}

}


#endif // ISTA_H
