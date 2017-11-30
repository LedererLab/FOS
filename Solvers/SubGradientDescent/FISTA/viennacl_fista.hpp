#ifndef VIENNACL_CL_FISTA_HPP
#define VIENNACL_CL_FISTA_HPP

// C System-Headers
//
// C++ System headers
//
// ViennCL Headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
// Project Specific Headers
#include "../../../Generic/generics.hpp"
#include "../viennacl_subgradient_descent.hpp"

namespace hdim {

template < typename T, typename Base = internal::CL_Solver< T > >
/*!
 * \brief Run the Fast Iterative Shrinking and Thresholding Algorthim.
 */
class CL_FISTA : public internal::CL_SubGradientSolver<T,Base> {

  public:
    CL_FISTA( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, T L_0 = 0.1 );

  protected:
    viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda );

  private:

    viennacl::vector<T> update_beta_fista (
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta,
        T L,
        T thres );

    viennacl::vector<T> y_k;
    viennacl::vector<T> y_k_old;

    viennacl::vector<T> x_k_less_1;

    const T eta = 1.5;
    T t_k = static_cast<T>( 1 );
    T L = static_cast<T>( 0 );
};

template < typename T, typename Base >
CL_FISTA<T,Base>::CL_FISTA( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, T L_0 ) : internal::CL_SubGradientSolver<T,Base>( L_0 ) {

    y_k = viennacl::vector<T>( Beta_0.rows() );
    y_k_old = viennacl::vector<T>( Beta_0.rows() );

    viennacl::copy( Beta_0, y_k );
    t_k = static_cast<T>( 1 );
}

template < typename T, typename Base >
viennacl::vector<T> CL_FISTA<T,Base>::update_beta_fista (
    const viennacl::matrix<T> &X,
    const viennacl::vector<T> &Y,
    const viennacl::vector<T> &Beta,
    T L,
    T thres ) {

    viennacl::vector<T> x_k = Beta;

    x_k_less_1 = x_k;
    x_k = internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, y_k, L, thres );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}

#ifdef DEBUG
template < typename T, typename Base >
viennacl::vector<T> CL_FISTA<T, Base>::update_rule(
    const viennacl::matrix<T> &X,
    const viennacl::vector<T> &Y,
    const viennacl::vector<T> &Beta,
    T lambda ) {

    L = internal::CL_SubGradientSolver<T,Base>::L_0;

    viennacl::copy( Beta, y_k );

//    y_k = Beta;

    y_k_old = y_k;

    viennacl::vector<T> y_k_temp = internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, y_k, L, lambda );

    unsigned int counter = 0;

    while( ( internal::CL_SubGradientSolver<T,Base>::f_beta( X, Y, y_k_temp ) > internal::CL_SubGradientSolver<T,Base>::f_beta_tilda( X, Y, y_k_temp, y_k_old, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;

        DEBUG_PRINT( "L: " << L );
        y_k_temp = internal::CL_SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return update_beta_fista( X, Y, Beta, L, lambda );;
}
#else
template < typename T, typename Base >
viennacl::vector<T> CL_FISTA<T,Base>::update_rule(
    const viennacl::matrix<T> &X,
    const viennacl::vector<T> &Y,
    const viennacl::vector<T> &Beta,
    T lambda ) {

//    L = internal::CL_SubGradientSolver<T,Base>::L_0;

//    y_k = Beta;

//    y_k_old = y_k;

//    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*y_k_old - Y ) );

//    Eigen::Matrix< T, Eigen::Dynamic, 1 > to_modify = y_k_old - (1.0/L)*f_grad;
//    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

//    unsigned int counter = 0;

//    T f_beta = ( X*y_k_temp - Y ).squaredNorm();

//    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_part = X*y_k_old - Y;
//    T taylor_term_0 = f_part.squaredNorm();

//    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = ( y_k_temp - y_k_old );

//    T taylor_term_1 = f_grad.transpose()*beta_diff;

//    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

//    T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    L = internal::CL_SubGradientSolver<T,Base>::L_0;

    viennacl::copy( Beta, y_k );

    y_k_old = y_k;

    viennacl::vector<T> f_grad =  2.0*viennacl::linalg::prod( viennacl::trans( X ), viennacl::linalg::prod( X, y_k_old ) - Y );
    viennacl::vector<T> to_modify = y_k_old - (1.0/L)*f_grad;

    viennacl::vector<T> thres_( 1 );
    viennacl::copy( std::vector<T> { lambda / L }, thres_ );

    viennacl::vector<T> y_k_temp ( y_k.size() );
    viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( to_modify, y_k_temp, thres_ ) );

    T f_beta = norm_sqr( static_cast<viennacl::vector<T>>(viennacl::linalg::prod( X, y_k_temp ) - Y ) );
    viennacl::vector<T> f_part = viennacl::linalg::prod( X, y_k_old ) - Y;

    T taylor_term_0 = norm_sqr( f_part );
    viennacl::vector<T> beta_diff = y_k_temp - y_k_old;

    T taylor_term_1 = viennacl::linalg::inner_prod( f_grad , beta_diff );

    T taylor_term_2 = (L/2.0)*norm_sqr( beta_diff );

    T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    while( f_beta > f_beta_tilde ) {

//        counter++;
//        DEBUG_PRINT( "Backtrace iteration: " << counter );

//        L*= eta;

//        DEBUG_PRINT( "L: " << L );
//        to_modify = y_k_old - (1.0/L)*f_grad;
//        y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

//        f_beta = ( X*y_k_temp - Y ).squaredNorm();;

//        beta_diff = ( y_k_temp - y_k_old );
//        taylor_term_1 = f_grad.transpose()*beta_diff;
//        taylor_term_2 = L/2.0*beta_diff.squaredNorm();

//        f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        L*= eta;

        to_modify = y_k_old - (1.0/L)*f_grad;

        viennacl::copy( std::vector<T> { lambda / L }, thres_ );
        viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( to_modify, y_k_temp, thres_ ) );

        f_beta = norm_sqr( static_cast<viennacl::vector<T>>(viennacl::linalg::prod( X, y_k_temp ) - Y ) );

        beta_diff = y_k_temp - y_k_old;

        taylor_term_1 = viennacl::linalg::inner_prod( f_grad, beta_diff );

        taylor_term_2 = (L/2.0)*norm_sqr( beta_diff );

        f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

    }

    //    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_k = Beta;

    //    x_k_less_1 = x_k;

    //    to_modify = y_k_old - (1.0/L)*f_grad;
    //    x_k = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

    //    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    //    t_k = t_k_plus_1;

    //    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    //    return x_k;

    viennacl::vector<T> x_k = Beta;

    x_k_less_1 = x_k;

    to_modify = y_k_old - (1.0/L)*f_grad;

    viennacl::copy( std::vector<T> { lambda / L }, thres_ );
    viennacl::ocl::enqueue( hdim::internal::CL_SubGradientSolver<T,Base>::soft_thres_kernel_->operator()( to_modify, x_k, thres_ ) );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}
#endif

}


#endif // VIENNACL_CL_FISTA_HPP
