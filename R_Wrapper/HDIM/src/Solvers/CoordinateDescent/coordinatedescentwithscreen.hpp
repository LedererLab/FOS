#ifndef COORDINATEDESCENTWITHSCREEN_HPP
#define COORDINATEDESCENTWITHSCREEN_HPP


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
#include "../../Screening/screening_rules.hpp"
#include "../solver.hpp"

namespace hdim {

template < typename T >
class CoordinateDescentWithScreen : public internal::Solver<T> {

  public:
    CoordinateDescentWithScreen( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~CoordinateDescentWithScreen();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
            const T lambda,
            const T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        const T lambda,
        const unsigned int num_iterations );

  private:
    std::vector<T> inverse_norms;

};

template < typename T >
CoordinateDescentWithScreen<T>::CoordinateDescentWithScreen(
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

template < typename T >
CoordinateDescentWithScreen<T>::~CoordinateDescentWithScreen() {}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CoordinateDescentWithScreen<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    const T lambda,
    const unsigned int num_iterations ) {

    const T lambda_half = lambda / 2.0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Residual = Y - X*Beta_0;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_A = X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_A = Beta;


    std::vector< unsigned int > active_set, inactive_set;

    // Initialize vector of values [ 0, 1, ..., p - 1, p ]
    std::vector< unsigned int > universe ( X.cols() );
    std::iota ( std::begin(universe), std::end(universe) , 0 );

    T duality_gap_2 = static_cast<T>( 0 );

    for( unsigned int j = 0; j < num_iterations ; j ++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > nu = DualPoint( X_A, Y, Beta_A, lambda_half );
        duality_gap_2 = DualityGap2( X_A, Y, Beta_A, nu, lambda_half );

        if( j % 10 == 0 ) {

            T radius = std::sqrt( 2.0 * duality_gap_2 / square( lambda ) );
            active_set = SafeActiveSet( X, nu, radius );

            X_A = slice( X, active_set );
            Beta_A = slice( Beta, active_set );

            std::set_difference( universe.begin(),
                                 universe.end(),
                                 active_set.begin(),
                                 active_set.end(),
                                 std::inserter(inactive_set, inactive_set.begin()) );
        }

        for( const auto& active_index : active_set ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > X_j = X.col( active_index );

            if( Beta( active_index ) != static_cast<T>(0) ) {
                Residual = Residual + X_j*Beta( active_index );
            }

            T inverse_norm_j = inverse_norms[ active_index ];

            T threshold = lambda_half * inverse_norm_j;
            T elem = inverse_norm_j*X_j.transpose()*Residual;

            Beta( active_index ) = soft_threshold<T>( elem, threshold );

            if( Beta( active_index ) != static_cast<T>(0) ) {
                Residual = Residual - X_j*Beta( active_index );
            }

        }

        for( const auto& inactive_index : inactive_set ) {
            Beta( inactive_index ) = 0.0;
        }

        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

    }

    return Beta;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CoordinateDescentWithScreen<T>::operator () (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    const T lambda,
    const T duality_gap_target ) {

    const T lambda_half = lambda / 2.0;
    const T dgt_half = duality_gap_target / 2.0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Residual = Y - X*Beta_0;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_A = X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_A = Beta;


    std::vector< unsigned int > active_set, inactive_set;

    // Initialize vector of values [ 0, 1, ..., p - 1, p ]
    std::vector< unsigned int > universe ( X.cols() );
    std::iota ( std::begin(universe), std::end(universe) , 0 );

    T duality_gap_2 = static_cast<T>( 0 );

    unsigned int counter = 0;

    do {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > nu = DualPoint( X_A, Y, Beta_A, lambda_half );
        duality_gap_2 = DualityGap2( X_A, Y, Beta_A, nu, lambda_half );

        if( counter % 10 == 0 ) {

            T radius = std::sqrt( 2.0 * duality_gap_2 / square( lambda ) );
            active_set = SafeActiveSet( X, nu, radius );

            X_A = slice( X, active_set );
            Beta_A = slice( Beta, active_set );

            std::set_difference( universe.begin(),
                                 universe.end(),
                                 active_set.begin(),
                                 active_set.end(),
                                 std::inserter(inactive_set, inactive_set.begin()) );
        }

        for( const auto& active_index : active_set ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > X_j = X.col( active_index );

            if( Beta( active_index ) != static_cast<T>(0) ) {
                Residual = Residual + X_j*Beta( active_index );
            }

            T inverse_norm_j = inverse_norms[ active_index ];

            T threshold = lambda_half * inverse_norm_j;
            T elem = inverse_norm_j*X_j.transpose()*Residual;

            Beta( active_index ) = soft_threshold<T>( elem, threshold );

            if( Beta( active_index ) != static_cast<T>(0) ) {
                Residual = Residual - X_j*Beta( active_index );
            }

        }

        for( const auto& inactive_index : inactive_set ) {
            Beta( inactive_index ) = 0.0;
        }

        DEBUG_PRINT( "Current Duality Gap: " << duality_gap_2 << " Current Target: " << dgt_half );
        DEBUG_PRINT( "Norm Squared of updated Beta: " << Beta.squaredNorm() );

        counter ++;

    } while ( duality_gap_2 > dgt_half );

    return Beta;
}

}

#endif // COORDINATEDESCENTWITHSCREEN_HPP
