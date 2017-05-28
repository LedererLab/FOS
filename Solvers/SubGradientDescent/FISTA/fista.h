#ifndef FISTA_H
#define FISTA_H

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
#include "../subgradient_descent.h"

namespace hdim {

template < typename T >
class FISTA : internal::SubGradientSolver<T> {

  public:
    FISTA();

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
    VectorT<T> update_rule(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta,
        T L,
        T lambda );

    const T eta = 1.5;

    VectorT<T> update_beta_fista (
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta,
        T L,
        T thres );

    VectorT<T> y_k;
    VectorT<T> y_k_old;

    MatrixT<T> x_k_less_1;

    T t_k = 1;
};

template < typename T >
FISTA<T>::FISTA() {
    static_assert(std::is_floating_point< T >::value, "FISTA can only be used with floating point types.");
}

template < typename T >
VectorT<T> FISTA<T>::update_beta_fista (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T thres ) {

    VectorT<T> x_k = Beta;

    x_k_less_1 = x_k;
    x_k = internal::SubGradientSolver<T>::update_beta_ista( X, Y, y_k, L, thres );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}

template < typename T >
VectorT<T> FISTA<T>::operator() (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T L_0,
    T lambda,
    uint num_iterations ) {

    T L = L_0;

    VectorT<T> Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    for( uint i = 0; i < num_iterations; i++ ) {

        update_rule( X, Y, Beta, L, lambda );

    }

    return Beta;

}

template < typename T >
VectorT<T> FISTA<T>::operator() (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T L_0,
    T lambda,
    T duality_gap_target ) {

    T L = L_0;

    VectorT<T> Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    do {

        Beta = update_rule( X, Y, Beta, L, lambda );

        DEBUG_PRINT( "Duality Gap:" << duality_gap( X, Y, Beta, lambda ) );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;

}

#ifdef DEBUG
template < typename T >
VectorT<T> FISTA<T>::update_rule(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T lambda ) {

    y_k_old = y_k;

    VectorT<T> y_k_temp = internal::SubGradientSolver<T>::update_beta_ista( X, Y, y_k, L, lambda );

    uint counter = 0;

    while( ( internal::SubGradientSolver<T>::f_beta( X, Y, y_k_temp ) > internal::SubGradientSolver<T>::f_beta_tilda( X, Y, y_k_temp, y_k_old, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;

        DEBUG_PRINT( "L: " << L );
        y_k_temp = internal::SubGradientSolver<T>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return update_beta_fista( X, Y, Beta, L, lambda );;
}
#else
template < typename T >
VectorT<T> FISTA<T>::update_rule(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T lambda ) {

    y_k_old = y_k;

    VectorT<T> f_grad = 2.0*( X.transpose()*( X*y_k_old - Y ) );

    VectorT<T> to_modify = y_k_old - (1.0/L)*f_grad;
    VectorT<T> y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

    uint counter = 0;

    T f_beta = ( X*y_k_temp - Y ).squaredNorm();

    VectorT<T> f_part = X*y_k_old - Y;
    T taylor_term_0 = f_part.squaredNorm();

    VectorT<T> beta_diff = ( y_k_temp - y_k_old );

    T taylor_term_1 = f_grad.transpose()*beta_diff;

    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

    T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    while( f_beta > f_beta_tilde ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;

        DEBUG_PRINT( "L: " << L );
        to_modify = y_k_old - (1.0/L)*f_grad;
        y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

        f_beta = ( X*y_k_temp - Y ).squaredNorm();;

        beta_diff = ( y_k_temp - y_k_old );
        taylor_term_1 = f_grad.transpose()*beta_diff;
        taylor_term_2 = L/2.0*beta_diff.squaredNorm();

        f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    }

    VectorT<T> x_k = Beta;

    x_k_less_1 = x_k;

    to_modify = y_k_old - (1.0/L)*f_grad;
    x_k = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}
#endif

}


#endif // FISTA_H
