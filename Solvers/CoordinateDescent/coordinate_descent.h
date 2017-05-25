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
#include "../../Generic/debug.h"
#include "../../Generic/generics.h"

namespace hdim {

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > CoordinateDescent (
    const MatrixT& X,
    const VectorT& Y,
    const VectorT& Beta_0,
    T lambda,
    T duality_gap_target ) {

    VectorT Beta = Beta_0;

    do {


        for( int i = 0; i < Beta.size() ; i++ ) {

            VectorT A_i = X.col( i );

            T threshold = lambda / A_i.squaredNorm();
            T inverse_norm = static_cast<T>( 1 )/( A_i.squaredNorm() );

            MatrixT A_negative_i = X;
            A_negative_i.col( i ) = VectorT::Zero( X.rows() );

            VectorT x_negative_i = Beta;
            x_negative_i( i ) = static_cast<T>( 0 );

            T residual =  inverse_norm*A_i.transpose()*( Y - A_negative_i*x_negative_i );
            Beta( i ) = soft_threshold<T>( residual, threshold );

        }

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << " Current Target: " << duality_gap_target );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

}

#endif // COORDINATE_DESCENT_H
