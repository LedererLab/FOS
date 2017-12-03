#ifndef ABSTRACTSOLVER_HPP
#define ABSTRACTSOLVER_HPP

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
#include "base_solver.hpp"

namespace hdim {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for all iterative solvers.
 *
 * This class supports two types of convergence criteria -- iterative and duality gap.
 */
class AbstractSolver : public BaseSolver < T > {

  protected:

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > update_rule(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda ) = 0;

};


}

}

#endif // ABSTRACTSOLVER_HPP
