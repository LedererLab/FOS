#ifndef COORDINATE_DESCENT_H
#define COORDINATE_DESCENT_H

// C System-Headers
//
// C++ System headers
//#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//
// Project Specific Headers
#include "../../Generic/debug.hpp"
#include "../../Generic/generics.hpp"

namespace hdim {

/*! \file
 *  \brief Coordinate Descent iterative solvers.
 */

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T >
/*!
\f{algorithm}{
  \caption{Coordinate Descent with duality gap convergence criteria}
  \begin{algorithmic}[1]
  \Statex
  \Input\tikzmark{k}
  \Statex $X \in \mathbb{R}^{n \times p} $ \Comment{ The design matrix }
  \Statex $Y \in \mathbb{R}^n$  \Comment{ The vector of predictors }
  \Statex $\beta \in \mathbb{R}^n$  \Comment{ Starting vector }
  \Statex $\lambda \in \mathbb{R}$  \Comment{ Grid element }
  \Statex $\mathcal{D} \in \mathbb{R}$  \Comment{ Duality gap target }\tikzmark{l}
    \State $\widetilde{\beta} \gets \beta$ \Comment{ Make a copy of $\beta$ }
      \Do
        \For{ $i \in 1, 2, \dots, p$ }
          \State $t \gets \frac{ \lambda }{ \LTwoSqr{ X_i } }$ \Comment{ Scale grid element by norm of the i'th column of design matrix }
          \State $X_{-i} \gets X_{ \forall j \neq i}$ \Comment{ Take all columns of design matrix not equal to $i$ }
          \State $\widetilde{\beta}_{-i} \gets \widetilde{\beta}_{ \forall j \neq i}$ \Comment{ Take all elements of predictors vectors not equal to $i$ }
          \State $r \gets \frac{ X_i^T\left( Y - X_{-i} \beta_{-i}\right) }{ \LTwoSqr{ X_i } }$ \Comment{ Compute the scaled residual }
          \State $\widetilde{\beta}_i \gets \frac{1}{2}\tau \left( 2 \times r, t \right)$ \Comment{ Update the i'th element of Beta }
        \EndFor
      \doWhile{ DG $( X, Y, \widetilde{\beta}, \lambda ) > \mathcal{D}$ }\\
  \end{algorithmic}
  \Return $\widetilde{\beta}$
 \f}
 */
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > CoordinateDescent (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T lambda,
    T duality_gap_target ) {

    VectorT<T> Beta = Beta_0;

    do {

        for( int i = 0; i < Beta.size() ; i++ ) {

            VectorT<T> X_i = X.col( i );
            T inverse_norm = static_cast<T>( 1 )/( X_i.squaredNorm() );

//            MatrixT<T> X_negative_i = X;
//            X_negative_i.col( i ) = VectorT<T>::Zero( X.rows() );

            VectorT<T> Beta_negative_i = Beta;
            Beta_negative_i( i ) = static_cast<T>( 0 );

            T threshold = lambda / ( 2.0*X_i.squaredNorm() );
            T elem = inverse_norm*X_i.transpose()*( Y - X*Beta_negative_i );

            Beta( i ) = soft_threshold<T>( elem, threshold );

        }

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << " Current Target: " << duality_gap_target );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > CoordinateDescentStandardized (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    T lambda,
    T duality_gap_target ) {

    VectorT<T> Beta = Beta_0;
    T X_rows = static_cast<T>( X.rows() );

    do {

        for( int i = 0; i < Beta.size() ; i++ ) {

            VectorT<T> X_i = X.col( i );
            T inverse_norm = static_cast<T>( 1 )/( X_rows );

//            MatrixT<T> X_negative_i = X;
//            X_negative_i.col( i ) = VectorT<T>::Zero( X.rows() );

            VectorT<T> Beta_negative_i = Beta;
            Beta_negative_i( i ) = static_cast<T>( 0 );

            T threshold = lambda / ( 2.0*X_rows );
            T elem = inverse_norm*X_i.transpose()*( Y - X*Beta_negative_i );

            Beta( i ) = soft_threshold<T>( elem, threshold );

        }

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << " Current Target: " << duality_gap_target );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

}

#endif // COORDINATE_DESCENT_H
