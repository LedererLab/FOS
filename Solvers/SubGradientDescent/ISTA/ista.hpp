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
#include "../../../Generic/generics.hpp"
#include "../../../Generic/debug.hpp"
#include "../subgradient_descent.hpp"

namespace hdim {

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T >
/*!
 * \brief Run the Iterative Shrinking and Thresholding Algorthim.
 */
class ISTA : public internal::SubGradientSolver<T> {

  public:

    /*!
    \f{algorithm}{
      \caption{ISTA with backtracking line search and iterative convergence criteria}
      \begin{algorithmic}[1]
      \Statex
      \Input\tikzmark{k}
      \Statex $X \in \mathbb{R}^{n \times p} $ \Comment{ The design matrix }
      \Statex $Y \in \mathbb{R}^n$  \Comment{ The vector of predictors }
      \Statex $\beta \in \mathbb{R}^n$  \Comment{ Starting vector }
      \Statex $L_0 \in \mathbb{R}$  \Comment{ Initial Lipschitz constant, used by backtracking line search }
      \Statex $\lambda \in \mathbb{R}$  \Comment{ Grid element }
      \Statex $\eta \in \mathbb{R}$  \Comment{ Step size when updating Lipschitz constant }
      \Statex $\mathcal{N} \in \mathbb{N}$  \Comment{ Number of times to run the algorithm }\tikzmark{l}
        \State $\widetilde{\beta} \gets \beta$ \Comment{ Make a copy of $\beta$ that will be updated during back tracking.}
          \For{$i \in \{ 1, 2, \dots, \mathcal{N} \}$}
            \State $ \widetilde{\beta} \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \widetilde{\beta}, L ) )$
            \While{ $f_{\beta} ( X, Y, \widetilde{\beta} ) > f_{\widetilde{\beta}}( X, Y, \widetilde{\beta}, \beta_, L) $ }
              \State $L \gets \eta L$
              \State $ \widetilde{\beta} \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \beta ) )$
            \EndWhile
            \State $\beta \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \beta, L ) )$ \Comment{ Update $\beta$ once $L$ is sufficiently large.}
          \EndFor
      \end{algorithmic}
      \Return $\beta$
     \f}
     */
    VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        T lambda,
        uint num_iterations );

    /*!
    \f{algorithm}{
      \caption{ISTA with backtracking line search and duality gap convergence criteria}
      \begin{algorithmic}[1]
      \Statex
      \Input\tikzmark{k}
      \Statex $X \in \mathbb{R}^{n \times p} $ \Comment{ The design matrix }
      \Statex $Y \in \mathbb{R}^n$  \Comment{ The vector of predictors }
      \Statex $\beta \in \mathbb{R}^n$  \Comment{ Starting vector }
      \Statex $L_0 \in \mathbb{R}$  \Comment{ Initial Lipschitz constant, used by backtracking line search }
      \Statex $\lambda \in \mathbb{R}$  \Comment{ Grid element }
      \Statex $\eta \in \mathbb{R}$  \Comment{ Step size when updating Lipschitz constant }
      \Statex $\mathcal{D} \in \mathbb{R}$  \Comment{ Duality gap target }\tikzmark{l}
        \State $\widetilde{\beta} \gets \beta$ \Comment{ Make a copy of $\beta$ that will be updated during back tracking.}
          \Do
            \State $ \widetilde{\beta} \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \widetilde{\beta}, L ) )$
            \While{ $f_{\beta} ( X, Y, \widetilde{\beta} ) > f_{\widetilde{\beta}}( X, Y, \widetilde{\beta}, \beta_, L) $ }
              \State $L \gets \eta L$
              \State $ \widetilde{\beta} \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \beta ) )$
            \EndWhile
            \State $\beta \gets \tau( \beta - \frac{1}{L} \nabla f( X, Y, \beta, L ) )$ \Comment{ Update $\beta$ once $L$ is sufficiently large.}
          \doWhile{ DG $( X, Y, \beta, \lambda ) > \mathcal{D}$ }\\
      \end{algorithmic}
      \Return $\beta$
     \f}
     */
    VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
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
VectorT<T> ISTA<T>::operator()(
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T lambda,
    uint num_iterations ) {

    T L = internal::SubGradientSolver<T>::L_0;

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
    T lambda,
    T duality_gap_target ) {

    T L = internal::SubGradientSolver<T>::L_0;

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

    return internal::SubGradientSolver<T>::update_beta_ista( X, Y, Beta, L, lambda );
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

    return ( Beta - 1.0/L*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );
}
#endif

}


#endif // ISTA_H
