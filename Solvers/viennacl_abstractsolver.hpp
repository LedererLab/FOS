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

namespace vcl {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for all iterative solvers.
 *
 * This class supports two types of convergence criteria -- iterative and duality gap.
 */
class AbstractSolver : public hdim::internal::BaseSolver < T > {

  public:

    virtual ~AbstractSolver() = 0;

    /*!
     * \brief Run the AbstractSolver for a fixed number of steps,
     * specified by num_iterations.
     *
     * \param X
     * An n x p design matrix.
     *
     * \param Y
     * A 1 x n vector of predictors.
     *
     * \param Beta_0
     * A 1 x n vector of starting parameters.
     *
     * \param lambda
     * Current grid element.
     *
     * \param num_iterations
     * The number of times the algorithm should iterate.
     *
     * \return
     * A 1 x n vector of results from the algorithm.
     */
    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations ) = 0;

    /*!
     * \brief Run the Sub-Gradient Descent algorithm until the duality gap is below
     * the threshold specified by duality_gap_target.
     *
     * \param X
     * An n x p design matrix.
     *
     * \param Y
     * A 1 x n vector of predictors.
     *
     * \param Beta_0
     * A 1 x n vector of starting parameters.
     *
     * \param lambda
     * Current grid element.
     *
     * \param duality_gap_target
     * The algorithm will iterate until the compute duality gap is below duality_gap_target.
     * Note care should be exercised, as the algorithm can iterate ad infinitum.
     *
     * \return
     * A 1 x n vector of results from the algorithm.
     */
    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target ) = 0;

  protected:

    virtual viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda ) = 0;

};

template < typename T >
AbstractSolver<T>::~AbstractSolver() {}

}

}

}

#endif // VIENNACL_ABSTRACTSOLVER_HPP
