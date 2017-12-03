#ifndef VIENNACL_SOLVER_HPP
#define VIENNACL_SOLVER_HPP

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
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/ocl/backend.hpp"
// OpenMP Headers
//
// Project Specific Headers
#include "../Generic/generics.hpp"
#include "../OpenCL_Generics/cl_generics.h"
#include "../Generic/debug.hpp"
#include "viennacl_abstractsolver.hpp"

namespace hdim {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for solvers that do not make use of GAP SAFE screening rules.
 */
class CL_Solver : public CL_AbstractSolver < T > {

  public:

    CL_Solver();
    virtual ~CL_Solver() = 0;

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

    virtual viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda ) = 0;

  private:

    T vcl_duality_gap ( const viennacl::matrix<T>& X,
                        const viennacl::vector<T>& Y,
                        const viennacl::vector<T>& Beta,
                        T r_stats_it );

    viennacl::matrix<T> X_;
    viennacl::vector<T> Y_;
    viennacl::vector<T> Beta_0_;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic > X_compare;
    Eigen::Matrix<T, Eigen::Dynamic, 1 > Y_compare;
    Eigen::Matrix<T, Eigen::Dynamic, 1 > Beta_0_compare;

    bool has_run = false;

};

template < typename T >
CL_Solver<T>::CL_Solver() {
    DEBUG_PRINT( "Using Plain CL_Solver.");

    // Use first available GPU as compute device
    viennacl::ocl::set_context_device_type( 0, viennacl::ocl::gpu_tag() );
    DEBUG_PRINT( "Current Context: " << viennacl::ocl::current_context().current_device().full_info() );

}

template < typename T >
CL_Solver<T>::~CL_Solver() {}

template < typename T >
T CL_Solver< T >::vcl_duality_gap ( const viennacl::matrix<T>& X,
                                 const viennacl::vector<T>& Y,
                                 const viennacl::vector<T>& Beta,
                                 T r_stats_it ) {

    //Computation of Primal Objective

    viennacl::vector<T> error = viennacl::linalg::prod( X, Beta ) - Y;
    T error_sqr_norm = norm_sqr( error );

    T f_beta = error_sqr_norm + r_stats_it* viennacl::linalg::norm_1( Beta );

    //Computation of Dual Objective

    //Compute dual point

    T alternative = r_stats_it / ( 2.0* viennacl::linalg::norm_inf( viennacl::linalg::prod( viennacl::trans( X ), error ) ) );
//    T alt_part_1 = viennacl::linalg::prod( viennacl::trans( Y ), error );
    T alt_part_1 = viennacl::linalg::inner_prod( Y, error );
    T alternative_0 = alt_part_1/( error_sqr_norm );

    T s = std::min( std::max( alternative, alternative_0 ), -alternative );

    viennacl::vector<T> nu_part = ( -2.0*s / r_stats_it ) * error + 2.0 / r_stats_it*  Y;

    T d_nu = 0.25*square( r_stats_it )* norm_sqr( nu_part ) - norm_sqr( Y );

    return f_beta + d_nu;
}

// Iterative
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CL_Solver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    if( !has_run ) {

        X_ = viennacl::matrix<T>( X.rows(), X.cols() );
        Y_ = viennacl::vector<T>( X.rows() );
        Beta_0_ = viennacl::vector<T>( X.cols() );

        X_compare = X;
        Y_compare = Y;
        Beta_0_compare = Beta_0;

        viennacl::copy( X, X_ );
        viennacl::copy( Y, Y_ );
        viennacl::copy( Beta_0, Beta_0_ );

    }

    if( X_compare != X ) {
        viennacl::copy( X, X_ );
        DEBUG_PRINT( "Value for X has changed!" );
    }
    if( Y_compare != Y ) {
        viennacl::copy( Y, Y_ );
        DEBUG_PRINT( "Value for Y has changed!" );
    }
    if( Beta_0_compare != Beta_0 ) {
        viennacl::copy( Beta_0, Beta_0_ );
        DEBUG_PRINT( "Value for Beta has changed!" );
    }

    viennacl::vector<T> Beta_ = Beta_0_;

    bool optim_continue = true;

    while( optim_continue ) {

        T dg = vcl_duality_gap( X_, Y_, Beta_, lambda );

        if( dg <= duality_gap_target ) {
            optim_continue = false;
        }

        Beta_ = update_rule( X_, Y_, Beta_, lambda );
    }

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta( Beta_0.rows(), 1 );
    viennacl::copy( Beta_, Beta );

    has_run |= true; //We've run at least once

    return Beta;

}

// Duality Gap Convergence Criteria
template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > CL_Solver<T>::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    unsigned int num_iterations ) {

    X_ = viennacl::matrix<T>( X.rows(), X.cols() );
    Y_ = viennacl::vector<T>( X.rows() );
    Beta_0_ = viennacl::vector<T>( X.cols() );

    viennacl::copy( X, X_ );
    viennacl::copy( Y, Y_ );
    viennacl::copy( Beta_0, Beta_0_ );

    viennacl::vector<T> Beta_ = Beta_0_;

    for( unsigned int i = 0; i < num_iterations ; i++ ) {

        Beta_ = update_rule( X_, Y_, Beta_, lambda );
        DEBUG_PRINT( "Duality Gap:" << vcl_duality_gap( X_, Y_, Beta_, lambda ) );

    }

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta ( Beta_0.rows(), 1 );
    viennacl::copy( Beta_, Beta );

    return Beta;

}

}

}

#endif // VIENNACL_SOLVER_HPP
