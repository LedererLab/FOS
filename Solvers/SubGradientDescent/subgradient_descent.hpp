#ifndef SUBGRADIENT_DESCENT_H
#define SUBGRADIENT_DESCENT_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// Boost Headers
//
// SPAMS Headers
//
// OpenMP Headers
//
// Project Specific Headers
#include "../../Generic/generics.hpp"
#include "../../Generic/debug.hpp"
#include "../solver.hpp"
#include "../screeningsolver.hpp"

namespace hdim {

namespace internal {

template < typename T, typename Base = internal::Solver<T> >

/*!
 * \brief Abstract base class for Sub-Gradient Descent algorithms
 * ,such as ISTA and FISTA, with backtracking line search.
 */
class SubGradientSolver : public Base {

  public:
    SubGradientSolver( T L = 0.1 );
    ~SubGradientSolver();

  protected:

    T f_beta (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    T f_beta_tilda (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
        T L );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_ista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    const T L_0;

};

template < typename T, typename Base >
SubGradientSolver< T, Base >::SubGradientSolver( T L ) : L_0( L ) {
    static_assert(std::is_floating_point< T >::value,\
                  "Subgradient descent methods can only be used with floating point types.");
}

template < typename T, typename Base >
SubGradientSolver< T, Base >::~SubGradientSolver() {}

template < typename T, typename Base >
T SubGradientSolver< T, Base >::f_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T, typename  Base >
T SubGradientSolver< T, Base >::f_beta_tilda (
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

template < typename T, typename Base >
Eigen::Matrix< T, Eigen::Dynamic, 1 > SubGradientSolver< T, Base >::update_beta_ista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

}

}

#endif // SUBGRADIENT_DESCENT_H
