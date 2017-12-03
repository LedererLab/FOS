#ifndef ISTA_H
#define ISTA_H

// C System-Headers
//
// C++ System headers
#include <functional>
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
#include "../../../Generic/debug.hpp"
#include "../subgradient_descent.hpp"

namespace hdim {

//template< typename T >
//using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

//template< typename T >
//using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T, typename Base = internal::Solver< T > >
/*!
 * \brief Run the Iterative Shrinking and Thresholding Algorthim.
 */
class ISTA : public internal::SubGradientSolver<T,Base> {

  public:
    ISTA( T L_0 = 0.1 );

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_rule(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda );

  private:
    const T eta = 1.5;
    T L = static_cast<T>( 0 );

};

template < typename T, typename Base >
ISTA<T,Base>::ISTA( T L_0 ) : internal::SubGradientSolver<T,Base>( L_0 ) {}

#if defined DEBUG
template < typename T, typename Base >
Eigen::Matrix< T, Eigen::Dynamic, 1 > ISTA<T,Base>::update_rule(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    unsigned int counter = 0;
    L = internal::SubGradientSolver<T,Base>::L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    counter++;
    DEBUG_PRINT( "Backtrace iteration: " << counter );

    while( ( internal::SubGradientSolver<T,Base>::f_beta( X, Y, Beta_temp ) > internal::SubGradientSolver<T,Base>::f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;
        Beta_temp = internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );
}
#else
template < typename T, typename Base >
Eigen::Matrix< T, Eigen::Dynamic, 1 > ISTA<T,Base>::update_rule(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T lambda ) {

    L = internal::SubGradientSolver<T,Base>::L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

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

    return ( Beta - 1.0/L*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );
}
#endif

}


#endif // ISTA_H
