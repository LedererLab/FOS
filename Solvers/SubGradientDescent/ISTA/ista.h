#ifndef ISTA_H
#define ISTA_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
//
// Boost Headers
//
// SPAMS Headers
//
// OpenMP Headers
//
// Project Specific Headers
#include "../../../Generic/generics.h"
#include "../../../Generic/debug.h"
#include "../subgradient_descent.h"

namespace hdim {

template < typename T >
class ISTA : public internal::SubGradientSolver<T> {

  public:
    ISTA();

    VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        T L_0,
        T lambda,
        uint num_iterations );

    VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target );

  private:
    VectorT<T> update_rule(const MatrixT<T>& X,
                           const VectorT<T>& Y,
                           const VectorT<T>& Beta,
                           T L_0,
                           T lambda );

    const T eta = 1.5;

};

template < typename T >
ISTA<T>::ISTA() {
    static_assert(std::is_floating_point< T >::value,\
                  "ISTA can only be used with floating point types.");
}

template < typename T >
VectorT<T> ISTA<T>::operator()(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T L_0,
    T lambda,
    uint num_iterations ) {

    T L = L_0;

    VectorT<T> Beta = Beta_0;

    for( uint i = 0; i < num_iterations; i++ ) {

        Beta = update_rule( X, Y, Beta, L, lambda );

    }

    return Beta;

}

template < typename T >
VectorT<T> ISTA<T>::operator()(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T L_0, \
    T lambda,
    T duality_gap_target ) {

    T L = L_0;

    VectorT<T> Beta = Beta_0;

    do {

        Beta = update_rule( X, Y, Beta, L, lambda );

        DEBUG_PRINT( "Duality Gap:" << duality_gap( X, Y, Beta, lambda ) );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;

}

#ifdef DEBUG
template < typename T >
VectorT<T> ISTA<T>::update_rule(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L_0,
    T lambda ) {

    uint counter = 0;
    T L = L_0;

    VectorT<T> Beta_temp = internal::SubGradientSolver<T>::update_beta_ista( X, Y, Beta, L, lambda );

    counter++;
    DEBUG_PRINT( "Backtrace iteration: " << counter );

    while( ( internal::SubGradientSolver<T>::f_beta( X, Y, Beta_temp ) > internal::SubGradientSolver<T>::f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;
        Beta_temp = internal::SubGradientSolver<T>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return internal::SubGradientSolver<T>::update_beta_ista( X, Y, Beta, L, lambda );;
}
#else
template < typename T >
VectorT<T> ISTA<T>::update_rule(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T lambda ) {

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

    return ( Beta - 1.0/L*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );;
}
#endif

}


#endif // ISTA_H
