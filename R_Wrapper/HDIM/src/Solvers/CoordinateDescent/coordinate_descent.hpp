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

namespace hdim {

template < typename T >
class CoordinateDescentSolver : public internal::Solver<T> {

  public:
    CoordinateDescentSolver(const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~CoordinateDescentSolver();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations );

  private:
    std::vector<T> thresholds;
    std::vector<T> p_1;
    std::vector< Eigen::Matrix< T, 1, Eigen::Dynamic > > p_2;

};

template < typename T >
CoordinateDescentSolver<T>::CoordinateDescentSolver(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0) {

    static_assert(std::is_floating_point< T >::value,\
                  "Coordinate Descent can only be used with floating point types.");

    thresholds.reserve ( Beta_0.size() );
    p_1.reserve( Beta_0.size() );
    p_2.reserve( Beta_0.size() );

    for( int i = 0; i < Beta_0.size() ; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > X_i = X.col( i );
        T inverse_norm = static_cast<T>( 1 )/( X_i.squaredNorm() );

        T threshold = 0.5*inverse_norm;

        T element_piece_1 = inverse_norm*X_i.transpose()*Y;
        Eigen::Matrix< T, 1, Eigen::Dynamic > element_piece_2 = inverse_norm*X_i.transpose()*X;

        thresholds.push_back( threshold );
        p_1.push_back( element_piece_1 );
        p_2.push_back( element_piece_2 );

    }

}

template < typename T >
CoordinateDescentSolver<T>::~CoordinateDescentSolver() {}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CoordinateDescentSolver<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    do {

        for( int i = 0; i < Beta.size() ; i++ ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_negative_i = Beta;
            Beta_negative_i( i ) = static_cast<T>( 0 );

            T elem = p_1.at( i ) - p_2.at( i )*Beta_negative_i;
            Beta( i ) = soft_threshold<T>( elem, lambda*thresholds.at( i ) );

        }

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << " Current Target: " << duality_gap_target );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CoordinateDescentSolver<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations) {

    (void)X;
    (void)Y;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    for( unsigned int j = 0; j < num_iterations; j ++) {

        for( int i = 0; i < Beta.size() ; i++ ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_negative_i = Beta;
            Beta_negative_i( i ) = static_cast<T>( 0 );

            T elem = p_1.at( i ) - p_2.at( i )*Beta_negative_i;
            Beta( i ) = soft_threshold<T>( elem, lambda*thresholds.at( i ) );

        }

        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    }

    return Beta;
}

template < typename T >
class LazyCoordinateDescent : public internal::Solver<T> {

  public:
    LazyCoordinateDescent(const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                          const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                          const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~LazyCoordinateDescent();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations );

  private:
    std::vector<T> inverse_norms;

};

template < typename T >
LazyCoordinateDescent<T>::LazyCoordinateDescent(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &Y,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &Beta_0 ) {

    (void)Y;

    inverse_norms.reserve( Beta_0.size() );

    for( int i = 0; i < Beta_0.size() ; i++ ) {

        T inverse_norm = static_cast<T>( 1 )/( X.col( i ).squaredNorm() );
        inverse_norms.push_back( inverse_norm );

    }
}

template < typename T >
LazyCoordinateDescent<T>::~LazyCoordinateDescent() {}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > LazyCoordinateDescent<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Residual = Y - X*Beta_0;

    for( unsigned int j = 0; j < num_iterations; j ++) {


        for( int i = 0; i < Beta.size() ; i++ ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > X_i = X.col( i );
//            T inverse_norm = static_cast<T>(1)/X_i.squaredNorm();
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

    }

    return Beta;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > LazyCoordinateDescent<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Residual = Y - X*Beta_0;


    do {

        for( int i = 0; i < Beta.size() ; i++ ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > X_i = X.col( i );
//            T inverse_norm = static_cast<T>(1)/X_i.squaredNorm();

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

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << " Current Target: " << duality_gap_target );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

}

#endif // COORDINATE_DESCENT_H
