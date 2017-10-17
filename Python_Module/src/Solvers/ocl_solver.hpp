#ifndef OCL_SOLVER_HPP
#define OCL_SOLVER_HPP

//Header for this file
//
//C System-Headers
//
//C++ System headers
//
//OpenCL Headers
//
//Boost Headers
//
//Eigen3 Headers
#include <eigen3/Eigen/Dense>
//Project specific headers
#include "../OpenCL_Base/openclbase.h"

namespace ocl {

namespace internal {

template< typename T >
/*!
 * \brief Abstract base class for Solvers
 *
 * This class supports two types of convergence criteria -- iterative and duality gap.
 */
class Solver : public OpenCLBase {
  public:

    Solver( uint platform_number = 0, uint device_number = 0 ) : OpenCLBase( platform_number, device_number ) {}
    virtual ~Solver() {}

    /*!
     * \brief Run the Solver for a fixed number of steps,
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
};

}

}

#endif // OCL_SOLVER_HPP
