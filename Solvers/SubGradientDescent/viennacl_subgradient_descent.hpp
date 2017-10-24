#ifndef VIENNACL_SUBGRADIENT_DESCENT_HPP
#define VIENNACL_SUBGRADIENT_DESCENT_HPP

// C System-Headers
//
// C++ System headers
//
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
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/ocl/program.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/enqueue.hpp"
// OpenMP Headers
//
// Project Specific Headers
#include "../../OpenCL_Generics/cl_generics.h"
#include "../../Generic/generics.hpp"
#include "../../Generic/debug.hpp"
#include "../viennacl_solver.hpp"
#include "../screeningsolver.hpp"

namespace hdim {

namespace vcl {

namespace internal {

template < typename T, typename Base = vcl::internal::Solver<T> >

/*!
 * \brief Abstract base class for Sub-Gradient Descent algorithms
 * ,such as ISTA and FISTA, with backtracking line search.
 */
class SubGradientSolver : public Base {

  public:
    SubGradientSolver( T L = 0.1 );
    ~SubGradientSolver();

  protected:

    T f_beta (
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta );

    T f_beta_tilda (
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta,
        const viennacl::vector<T>& Beta_prime,
        T L );

    viennacl::vector<T> update_beta_ista (
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta,
        T L,
        T thres );

    const std::string softthreshold_kernel = R"END(

       __kernel void SoftThreshold( __global float* input, __global float* output, __global float* threshold )
       {

           int i = get_global_id(0);

           float X_i_j = input[i];
           float signum = (float)( X_i_j >= 0.0 );

           float fragment = fabsf( X_i_j ) - threshold[0];
           float pos_part = ( fragment >= 0.0 )?( fragment ):( 0.0 );

           output[i] = signum*pos_part;

       }

  )END";

    viennacl::ocl::program* program_;
    viennacl::ocl::kernel* soft_thres_kernel_;

    const T L_0;

};

template < typename T, typename Base >
SubGradientSolver< T, Base >::SubGradientSolver( T L ) : L_0( L ) {
    static_assert(std::is_floating_point< T >::value,\
                  "Subgradient descent methods can only be used with floating point types.");

    program_ = &viennacl::ocl::current_context().add_program(softthreshold_kernel.c_str(), "softthreshold_kernel");
    soft_thres_kernel_ = &program_->get_kernel("SoftThreshold");
}

template < typename T, typename Base >
SubGradientSolver< T, Base >::~SubGradientSolver() {}

template < typename T, typename Base >
T SubGradientSolver< T, Base >::f_beta (
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta ) {

    viennacl::vector<T> scratch = viennacl::linalg::prod( X, Beta ) - Y;
    return norm_sqr( scratch );

}

template < typename T, typename  Base >
T SubGradientSolver< T, Base >::f_beta_tilda (
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta,
    const viennacl::vector<T>& Beta_prime,
    T L ) {

    viennacl::vector<T> f_beta = viennacl::linalg::prod( X,  Beta_prime ) - Y;
    T taylor_term_0 = norm_sqr( f_beta );

    viennacl::vector<T> f_grad = 2.0*viennacl::linalg::prod( viennacl::trans( X ),  f_beta );
    viennacl::vector<T> beta_diff = ( Beta - Beta_prime );

//    T taylor_term_1 = viennacl::linalg::prod( viennacl::trans( f_grad ), beta_diff );
    T taylor_term_1 = viennacl::linalg::inner_prod( f_grad, beta_diff );

    T taylor_term_2 = L/2.0*norm_sqr( beta_diff );

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T, typename Base >
viennacl::vector<T> SubGradientSolver< T, Base >::update_beta_ista (
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta,
    T L,
    T thres ) {

    viennacl::vector<T> thres_ { thres/L };

    viennacl::vector<T> f_beta = viennacl::linalg::prod( X, Beta ) - Y;

    viennacl::vector<T> f_grad = 2.0*viennacl::linalg::prod( viennacl::trans( X ), f_beta );
    viennacl::vector<T> beta_to_modify = Beta - (1.0/L)*f_grad;
    viennacl::vector<T> beta_output = beta_to_modify;

    viennacl::ocl::enqueue( soft_thres_kernel_->operator()( beta_to_modify, beta_output, thres_ ) ) ;

    return beta_output;

//    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

}

}

}

#endif // VIENNACL_SUBGRADIENT_DESCENT_HPP
