#ifndef VIENNACL_CL_ISTA_HPP
#define VIENNACL_CL_ISTA_HPP

// C System-Headers
//
// C++ System headers
#include <functional>
// Eigen Headers
//
// Boost Headers
//
// ViennCL Headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
// OpenMP Headers
//
// Project Specific Headers
#include "../../../Generic/generics.hpp"
#include "../../../Generic/debug.hpp"
#include "../viennacl_subgradient_descent.hpp"

namespace hdim {

template < typename T, typename Base = internal::CL_Solver< T > >
/*!
 * \brief Run the Iterative Shrinking and Thresholding Algorthim.
 */
class CL_ISTA : public internal::CL_SubGradientSolver<T,Base> {

  public:
    CL_ISTA( T L_0 = 0.1 );

  protected:
    viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda );

  private:
    const T eta = 1.5;
    T L = static_cast<T>( 0 );

};

template < typename T, typename Base >
CL_ISTA<T,Base>::CL_ISTA( T L_0 ) : internal::CL_SubGradientSolver<T,Base>( L_0 ) {}

#ifdef DEBUG
template < typename T, typename Base >
viennacl::vector<T> CL_ISTA<T,Base>::update_rule(
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta_0,
    T lambda ) {

    viennacl::vector<T> Beta = Beta_0;

    unsigned int counter = 0;
    L = internal::CL_SubGradientSolver<T,Base>::L_0;

    viennacl::vector<T> Beta_temp = internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    counter++;
    DEBUG_PRINT( "Backtrace iteration: " << counter );

    while( ( internal::CL_SubGradientSolver<T,Base>::f_beta( X, Y, Beta_temp ) > internal::CL_SubGradientSolver<T,Base>::f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;
        Beta_temp = internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );
}

#else
template < typename T, typename Base >
viennacl::vector<T> CL_ISTA<T,Base>::update_rule(
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta,
    T lambda ) {

    L = internal::CL_SubGradientSolver<T,Base>::L_0;

    // Extraordinarily silly way to pass a single value to the proximal operator kernel...
    viennacl::vector<T> thres_( 1 );
    viennacl::copy( std::vector<T> { lambda / L }, thres_ );

    viennacl::vector<T> f_grad = 2.0*viennacl::linalg::prod( viennacl::trans( X ), viennacl::linalg::prod( X, Beta ) - Y );
    viennacl::vector<T> beta_to_modify = Beta - (1.0/L)*f_grad;
    viennacl::vector<T> beta_temp = beta_to_modify;

    viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( beta_to_modify, beta_temp, thres_ ) );

    T f_beta = norm_sqr( static_cast<viennacl::vector<T>>( viennacl::linalg::prod( X, beta_temp ) - Y ) );

    viennacl::vector<T> f_part = viennacl::linalg::prod( X, Beta ) - Y;
    T taylor_term_0 = norm_sqr( f_part );

    viennacl::vector<T> beta_diff = beta_temp - Beta;

    T taylor_term_1 = viennacl::linalg::inner_prod( f_grad , beta_diff );
    T taylor_term_2 = ( L / 2.0 )*norm_sqr( beta_diff );

    T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    while( f_beta > f_beta_tilde ) {

        L*= eta;

        beta_to_modify = Beta - (1.0/L)*f_grad;

        viennacl::copy( std::vector<T> { lambda / L }, thres_ );
        viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( beta_to_modify, beta_temp, thres_ ) );

        f_beta = norm_sqr( static_cast<viennacl::vector<T>>( viennacl::linalg::prod( X, beta_temp ) - Y ) );

        beta_diff = beta_temp - Beta;

        taylor_term_1 = viennacl::linalg::inner_prod( f_grad , beta_diff );
        taylor_term_2 = ( L / 2.0 )*norm_sqr( beta_diff );

        f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    }

    beta_to_modify = Beta - (1.0/L)*f_grad;
    viennacl::copy( std::vector<T> { lambda / L }, thres_ );

    viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( beta_to_modify, beta_temp, thres_ ) );

    return beta_temp;

}
#endif

}

#endif // VIENNACL_CL_ISTA_HPP
