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
#include "../../../Generic/generics.hpp"
#include "../subgradient_descent.hpp"

namespace hdim {

template < typename T >
class FISTA : public internal::SubGradientSolver<T> {

  public:
    FISTA( T L_0 = 0.1 );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

  private:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_rule(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L_0,
        T lambda );

    const T eta = 1.5;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_fista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_old;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x_k_less_1;
    T t_k = static_cast<T>( 1 );

    T L = static_cast<T>( 0 );
};

template < typename T >
FISTA<T>::FISTA( T L_0 ) : internal::SubGradientSolver<T>( L_0 ) {}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > FISTA<T>::update_beta_fista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_k = Beta;

    x_k_less_1 = x_k;
    x_k = internal::SubGradientSolver<T>::update_beta_ista( X, Y, y_k, L, thres );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > FISTA<T>::operator() (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations ) {

    T L = internal::SubGradientSolver<T>::L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    for( unsigned int i = 0; i < num_iterations; i++ ) {

        update_rule( X, Y, Beta, L, lambda );

    }

    return Beta;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > FISTA<T>::operator() (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    do {

        Beta = update_rule( X, Y, Beta, internal::SubGradientSolver<T>::L_0, lambda );

        DEBUG_PRINT( "Duality Gap:" << duality_gap( X, Y, Beta, lambda ) );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;

}

#ifdef DEBUG
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > FISTA<T>::update_rule(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L_0,
    T lambda ) {

    L = L_0;

    y_k_old = y_k;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_temp = internal::SubGradientSolver<T>::update_beta_ista( X, Y, y_k, L, lambda );

    unsigned int counter = 0;

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
Eigen::Matrix< T, Eigen::Dynamic, 1 > FISTA<T>::update_rule(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L_0,
    T lambda ) {

    L = L_0;

    y_k_old = y_k;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*y_k_old - Y ) );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > to_modify = y_k_old - (1.0/L)*f_grad;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

    unsigned int counter = 0;

    T f_beta = ( X*y_k_temp - Y ).squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_part = X*y_k_old - Y;
    T taylor_term_0 = f_part.squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = ( y_k_temp - y_k_old );

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

    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_k = Beta;

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
