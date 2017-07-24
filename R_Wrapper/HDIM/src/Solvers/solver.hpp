#ifndef SOLVER_HPP
#define SOLVER_HPP

// C System-Headers
//
// C++ System headers
#include <functional> // std::function
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
#include "abstractsolver.hpp"

namespace hdim {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for solvers that do not make use of GAP SAFE screening rules.
 */
class Solver : public AbstractSolver < T > {

  public:

    Solver();
    virtual ~Solver() = 0;

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
Solver<T>::Solver() {
    DEBUG_PRINT( "Using Plain Solver.");
}

template < typename T >
Solver<T>::~Solver() {}

// Iterative
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > Solver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    bool optim_continue = true;

    while( optim_continue ) {

        T dg = duality_gap( X, Y, Beta, lambda );

        if( dg <= duality_gap_target ) {
            optim_continue = false;
        }

        Beta = update_rule( X, Y, Beta, lambda );
    }

    return Beta;

}

// Duality Gap Convergence Criteria
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > Solver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    for( unsigned int i = 0; i < num_iterations ; i++ ) {

        Beta = update_rule( X, Y, Beta, lambda );
        DEBUG_PRINT( "Duality Gap:" << duality_gap( X, Y, Beta, lambda ) );

    }

    return Beta;

}

}

}

#endif // SOLVER_HPP
