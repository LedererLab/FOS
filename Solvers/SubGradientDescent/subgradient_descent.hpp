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
#include "../../Generic/generics.hpp"

namespace hdim {

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for Sub-Gradient Descent algorithms
 * ,such as ISTA and FISTA, with backtracking line search.
 *
 * This class supports two types of convergence criteria -- iterative and duality gap.
 *
 * The iterative criteria will run the algorithm will run u
 */
class SubGradientSolver {

  public:
    SubGradientSolver();

    /*!
     * \brief Run the Sub-Gradient Descent algorithm for a fixed number of steps,
     * specified by num_iterations.
     *
     * \param X
     * An n x p design matrix.
     *
     * \param Y
     * A 1 x n vector of predictors.
     *
     * \param Beta_0
     * A 1 x n vector of starting parameters.
     *
     * \param L_0
     * The initial Lipschitz constant as used by backtracking line-search.
     *
     * \param lambda
     * Current grid element.
     *
     * \param num_iterations
     * The number of times the algorithm should iterate.
     *
     * \return
     * A 1 x n vector of results from the algorithm.
     */
    virtual VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        T L_0,
        T lambda,
        uint num_iterations ) = 0;

    /*!
     * \brief Run the Sub-Gradient Descent algorithm until the duality gap is below
     * the threshold specified by duality_gap_target.
     *
     * \param X
     * An n x p design matrix.
     *
     * \param Y
     * A 1 x n vector of predictors.
     *
     * \param Beta_0
     * A 1 x n vector of starting parameters.
     *
     * \param L_0
     * The initial Lipschitz constant as used by backtracking line-search.
     *
     * \param lambda
     * Current grid element.
     *
     * \param duality_gap_target
     * The algorithm will iterate until the compute duality gap is below duality_gap_target.
     * Note care should be exercised, as the algorithm can iterate ad infinitum.
     *
     * \return
     * A 1 x n vector of results from the algorithm.
     */
    virtual VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target ) = 0;

  protected:

    T f_beta (
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta );

    T f_beta_tilda (
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta,
        const VectorT<T>& Beta_prime,
        T L );

    VectorT<T> update_beta_ista (
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta,
        T L,
        T thres );
};

template < typename T >
SubGradientSolver<T>::SubGradientSolver() {
    static_assert(std::is_floating_point< T >::value,\
                  "Subgradient descent methods can only be used with floating point types.");
}

template < typename T >
T SubGradientSolver<T>::f_beta (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T >
T SubGradientSolver<T>::f_beta_tilda (
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
VectorT<T> SubGradientSolver<T>::update_beta_ista (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T thres ) {

    VectorT<T> f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    VectorT<T> beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

}

}

#endif // SUBGRADIENT_DESCENT_H
