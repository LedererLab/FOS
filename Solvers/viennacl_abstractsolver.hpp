#ifndef VIENNACL_ABSTRACTSOLVER_HPP
#define VIENNACL_ABSTRACTSOLVER_HPP

// C System-Headers
//
// C++ System headers
#include <functional> // std::function
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// Boost Headers
//
// ViennCL Headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
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
class CL_AbstractSolver : public BaseSolver < T > {

  protected:

    virtual viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda ) = 0;

};

}

}

#endif // VIENNACL_ABSTRACTSOLVER_HPP
