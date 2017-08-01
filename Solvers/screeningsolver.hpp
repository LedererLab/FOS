#ifndef SCREENINGSOLVER_HPP
#define SCREENINGSOLVER_HPP

// C System-Headers
//
// C++ System headers
#include <numeric>
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
#include "../Generic/generics.hpp"
#include "../Generic/debug.hpp"
#include "../Screening/screening_rules.hpp"
#include "abstractsolver.hpp"

namespace hdim {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for Solvers that use GAP SAFE screening rules.
 */
class ScreeningSolver : public AbstractSolver < T >  {

  public:

    ScreeningSolver();
    virtual ~ScreeningSolver() = 0;

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations );

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

  protected:

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > update_rule(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda ) = 0;

};

template < typename T >
ScreeningSolver<T>::ScreeningSolver() {
    DEBUG_PRINT( "Using Screening Rules" );
}

template < typename T >
ScreeningSolver<T>::~ScreeningSolver() {}

// Iterative
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > ScreeningSolver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    const T lambda_half = lambda / 2.0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_A = X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_A = Beta;

    std::vector< unsigned int > active_set, inactive_set;

    // Initialize vector of values [ 0, 1, ..., p - 1, p ]
    std::vector< unsigned int > universe ( X.cols() );
    std::iota ( std::begin(universe), std::end(universe) , 0 );

    T duality_gap_2 = static_cast<T>( 0 );

    bool optim_continue = true;
    unsigned int counter = 0;

    while( optim_continue ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > nu = DualPoint( X_A, Y, Beta_A, lambda_half );
        duality_gap_2 = DualityGap2( X_A, Y, Beta_A, nu, lambda_half );

        if( duality_gap_2 <= duality_gap_target ) {
            optim_continue = false;
        }

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

        Beta_A = update_rule( X_A, Y, Beta_A, lambda );

        for( unsigned int x = 0; x < active_set.size() ; x++ ) {

            unsigned int active_index = active_set[ x ];
            Beta( active_index ) = Beta_A[ x ];

        }

        for( const auto& inactive_index : inactive_set ) {
            Beta( inactive_index ) = 0.0;
        }

        counter ++;
        DEBUG_PRINT( "Duality Gap:" << duality_gap_2 );

    }

    return Beta;

}

// Duality Gap Convergence Criteria
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > ScreeningSolver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations ) {

    const T lambda_half = lambda / 2.0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_A = X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_A = Beta;


    std::vector< unsigned int > active_set, inactive_set;

    // Initialize vector of values [ 0, 1, ..., p - 1, p ]
    std::vector< unsigned int > universe ( X.cols() );
    std::iota ( std::begin(universe), std::end(universe) , 0 );

    T duality_gap_2 = static_cast<T>( 0 );

    for( unsigned int i = 0; i < num_iterations ; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > nu = DualPoint( X_A, Y, Beta_A, lambda_half );
        duality_gap_2 = DualityGap2( X_A, Y, Beta_A, nu, lambda_half );

        if( i % 10 == 0 ) {

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

        Beta_A = update_rule( X_A, Y, Beta_A, lambda );

        for( unsigned int x = 0; x < active_set.size() ; x++ ) {

            unsigned int active_index = active_set[ x ];
            Beta( active_index ) = Beta_A[ x ];

        }

        for( const auto& inactive_index : inactive_set ) {
            Beta( inactive_index ) = 0.0;
        }

        DEBUG_PRINT( "Duality Gap:" << duality_gap_2 );

    }

    return Beta;

}

}

}

#endif // SCREENINGSOLVER_HPP
