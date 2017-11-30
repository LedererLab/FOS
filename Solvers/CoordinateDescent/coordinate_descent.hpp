#ifndef COORDINATE_DESCENT_H
#define COORDINATE_DESCENT_H

// C System-Headers
//
// C++ System headers
#include <vector>
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
#include "../solver.hpp"
#include "../screeningsolver.hpp"

namespace hdim {

template < typename T, typename Base = internal::Solver<T> >
class LazyCoordinateDescent : public Base {

  public:
    LazyCoordinateDescent( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                           const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                           const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~LazyCoordinateDescent();

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_rule(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda );

  private:
    std::vector<T> inverse_norms;

};

template < typename T, typename Base >
LazyCoordinateDescent<T,Base>::LazyCoordinateDescent(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &Y,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &Beta_0 ) {

    (void)Y;

    inverse_norms.reserve( Beta_0.size() );

    for( int i = 0; i < Beta_0.size() ; i++ ) {

        T X_i_norm = X.col( i ).squaredNorm();

        T inverse_norm = ( X_i_norm == 0 )?( 0.0 ):( static_cast<T>(1)/X_i_norm );
        inverse_norms.push_back( inverse_norm );

    }
}

template < typename T, typename Base >
LazyCoordinateDescent<T,Base>::~LazyCoordinateDescent() {}

template < typename T, typename Base >
Eigen::Matrix< T, Eigen::Dynamic, 1 > LazyCoordinateDescent<T,Base>::update_rule(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Residual = Y - X*Beta_0;


    for( int i = 0; i < Beta.size() ; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > X_i = X.col( i );
        T inverse_norm = inverse_norms[i];

        if( Beta( i ) != static_cast<T>(0) ) {
            Residual = Residual + X_i*Beta( i );
        }


        T threshold = lambda / ( static_cast<T>(2) ) * inverse_norm;
        T elem = inverse_norm*X_i.transpose()*Residual;

        Beta( i ) = soft_threshold<T>( elem, threshold );

        if( Beta( i ) != static_cast<T>(0) ) {
            Residual = Residual - X_i*Beta( i );
        }

    }

    DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    return Beta;
}

}

#endif // COORDINATE_DESCENT_H
