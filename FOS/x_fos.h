#ifndef X_FOS_H
#define X_FOS_H


// C System-Headers
//
// C++ System headers
//#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//
// Project Specific Headers
#include "../Generic/debug.h"
#include "../Generic/generics.h"

namespace hdim {

namespace experimental {

template < typename T >
/*!
 * \brief The FOS algorithim
 */
class X_FOS {

  public:
    X_FOS();

    /*!
     * \brief Run the main X_FOS algorithm
     *
     * Calling this function will run the X_FOS algorithm using the values of
     * X and Y.
     *
     */
    void operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >&x,
                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >&y );

    T ReturnLambda();
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
    uint ReturnOptimIndex();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
    Eigen::Matrix< int, Eigen::Dynamic, 1 > ReturnSupport();

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > fos_fit;
    T lambda;
    uint optim_index;

  private:
    std::vector< T > GenerateLambdaGrid (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        uint M );

    bool ComputeStatsCond(T C,
                          uint stats_it,
                          T r_stats_it,
                          const std::vector<T> &lambdas,
                          const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas );

    T DualityGapTarget( uint r_stats_it );
    T duality_gap_target( T gamma, T C, T r_stats_it, uint n );

    T primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                        T r_stats_it );

    T dual_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                      const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                      const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                      T r_stats_it );

    T duality_gap ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                    T r_stats_it );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA_OPT (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > FISTA (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > FISTA_OPT (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T L_0, \
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > CoordinateDescent (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_ista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_fista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_cd (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    T f_beta_tilda (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
        T L );

    T f_beta (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_old;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x_k_less_1;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;

    const T C = 0.75;
    const uint M = 100;
    const T gamma = 1;

    bool statsCont = true;

    uint loop_index = 0;

    uint statsIt = 1;
    std::vector< T > lambda_grid;

    T t_k = 1;
    T hot_start_L = 0.1;
    const T L_0 = 0.1;

    int n = 1, p = 1;

};

template< typename T >
/*!
 * \brief Initialize a new algorithm, and instantiate member attributes X and Y.
 *
 * \param x
 * An n x p design matrix
 *
 * \param y
 * An n x 1 vector
 */
X_FOS< T >::X_FOS() {
    static_assert(std::is_floating_point< T >::value, "X_FOS can only be used with floating point types.");
}

template < typename T >
T X_FOS< T >::ReturnLambda() {
    return lambda;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS< T >::ReturnBetas() {
    return Betas;
}

template < typename T >
uint X_FOS< T >::ReturnOptimIndex() {
    return optim_index;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS< T >::ReturnCoefficients() {
    return fos_fit;
}

template < typename T >
Eigen::Matrix<int, Eigen::Dynamic, 1> X_FOS<T>::ReturnSupport() {

    T n_t = static_cast<T>( n );

    T cut_off = static_cast<T>( 6 )*C*lambda/n_t;
    std::cout << "Cut-off for Support Computation: " << cut_off << std::endl;

    return GenerateSupport( fos_fit, cut_off );

//    return fos_fit.unaryExpr( SupportSift<T>( C_t, lambda, n_t ) );

}

//Member functions

template < typename T >
/*!
 * \brief Determine the 'stop' condition for the outer loop
 *
 * \param stats_it
 * \param r_stats_it
 * \param lambdas
 * \return True if outer loop should continue, false otherwise
 */
bool X_FOS<T>::ComputeStatsCond( T C,
                                 uint stats_it,
                                 T r_stats_it,
                                 const std::vector < T >& lambdas,
                                 const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas ) {

    bool stats_cond = true;

    for ( uint i = 1; i <= stats_it; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_k = Betas.col( i - 1 );
        T rk = lambdas.at( i - 1 );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = Betas.col( stats_it - 1 ) - beta_k;
        T abs_max_betas = beta_diff.template lpNorm< Eigen::Infinity >();

        T n_t = static_cast<T>( n );
        T check_parameter = n_t*abs_max_betas / ( r_stats_it + rk );

        stats_cond &= ( check_parameter <= C );
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
std::vector< T > X_FOS<T>::GenerateLambdaGrid (

    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    uint M ) {

    T rMax = 2.0*( X.transpose() * Y ).template lpNorm< Eigen::Infinity >();
    T rMin = 0.001*rMax;

    return LogScaleVector( rMax, rMin, M );

}

template < typename T >
/*!
 * \brief Compute the target for the duality gap used in the inner loop
 *
 * The duality gap should be less than or equal to this target in order to
 * exit the inner loop.
 *
 * \param r_stats_it
 *
 * \return Target quantity
 */
T X_FOS<T>::DualityGapTarget( uint r_stats_it ) {
    T r_stats_it_f = static_cast<T>( r_stats_it );
    return square(C) * square( r_stats_it_f ) / static_cast<T>( n );
}

template < typename T >
inline T X_FOS<T>::duality_gap ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                                 T r_stats_it ) {

    //Computation of Primal Objective

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = X*Beta - Y;
    T error_sqr_norm = error.squaredNorm();

    T f_beta = error_sqr_norm + r_stats_it*Beta.template lpNorm < 1 >();

    //Computation of Dual Objective

    //Compute dual point

    T alternative = r_stats_it /( ( 2.0*X.transpose()*error ).template lpNorm< Eigen::Infinity >() );
    T alt_part_1 = static_cast<T>( Y.transpose()*error );
    T alternative_0 = alt_part_1/( error_sqr_norm );

    T s = std::min( std::max( alternative, alternative_0 ), -alternative );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_part = ( - 2.0*s / r_stats_it ) * error + 2.0/r_stats_it*Y;

    T d_nu = 0.25*square( r_stats_it )*nu_part.squaredNorm() - Y.squaredNorm();

    return f_beta + d_nu;
}

template < typename T >
T X_FOS< T >::primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                                T r_stats_it ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = Y - X*Beta;
    T f_beta = error.squaredNorm() + r_stats_it*Beta.template lpNorm < 1 >();

    return f_beta;

}

template < typename T >
T X_FOS< T >::dual_objective ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                               T r_stats_it ) {

    //Computation of s
    T s_chunk =  r_stats_it / ( 2.0*X.transpose()*( X*Beta - Y ) ).template lpNorm< Eigen::Infinity >();
    T s_chunk_prime = ( - static_cast<T>( Y.transpose()*( X*Beta - Y ) ) )/( Y - X*Beta ).squaredNorm();
    T s = std::min( std::max( - s_chunk, s_chunk_prime ), s_chunk );

    //Computation of nu tilde
    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_tilde = 2.0*s/r_stats_it*( X*Beta - Y );

    T d_nu = square( r_stats_it )/4.0*( nu_tilde + 2.0/r_stats_it*Y ).squaredNorm() - Y.squaredNorm();

    return d_nu;
}

template < typename T >
T X_FOS<T>::duality_gap_target( T gamma, T C, T r_stats_it, uint n ) {

    T n_f = static_cast<T>( n );
    return gamma*square( C )*square( r_stats_it )/n_f;

}

template < typename T >
T X_FOS<T>::f_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T >
T X_FOS<T>::f_beta_tilda (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
    T L ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_beta = X*Beta_prime - Y;
    T taylor_term_0 = f_beta.squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_grad = 2.0*X.transpose()*( f_beta );
    Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta - Beta_prime );

    T taylor_term_1 = f_grad.transpose()*beta_diff;

    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

//    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_grad = 2.0*X.transpose()*( f_beta );
//    T taylor_term_1 = static_cast<T>( f_grad.transpose()*Beta ) - static_cast<T>( f_grad.transpose()*Beta_prime );
//    T taylor_term_2 = L/2.0*( std::abs( Beta.squaredNorm() - Beta_prime.squaredNorm() ) );

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 >  X_FOS<T>::update_beta_ista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {


    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

//    return ( Beta + 2.0/L*(  X.transpose()*Y - X.transpose()*( X*Beta ) ) ).unaryExpr( SoftThres<T>( thres/L ) );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS<T>::update_beta_fista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_k = Beta;

    x_k_less_1 = x_k;
    x_k = update_beta_ista( X, Y, y_k, L, thres );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
           // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}


template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS<T>::FISTA_OPT (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T L_0, \
    T lambda,
    T duality_gap_target ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    uint outer_counter = 0;

    do {

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        y_k_old = y_k;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*y_k_old - Y ) );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > to_modify = y_k_old - (1.0/L)*f_grad;
        Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

        uint counter = 0;

        T f_beta = ( X*y_k_temp - Y ).squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > f_part = X*y_k_old - Y;
        T taylor_term_0 = f_part.squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( y_k_temp - y_k_old );

        T taylor_term_1 = f_grad.transpose()*beta_diff;

        T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

        T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        while( f_beta > f_beta_tilde ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;

            DEBUG_PRINT( "L: " << L );
            to_modify = y_k_old - (1.0/L)*f_grad;
            y_k_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

            f_beta = ( X*y_k_temp - Y ).squaredNorm();;

            beta_diff = ( y_k_temp - y_k_old );
            taylor_term_1 = f_grad.transpose()*beta_diff;
            taylor_term_2 = L/2.0*beta_diff.squaredNorm();

            f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        }

        Eigen::Matrix< T, Eigen::Dynamic, 1 > x_k = Beta;

        x_k_less_1 = x_k;

        to_modify = y_k_old - (1.0/L)*f_grad;
        x_k = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

        T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
        t_k = t_k_plus_1;

        y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

        Beta = x_k;

    } while ( ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target ));

    return Beta;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS<T>::FISTA (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T L_0, \
    T lambda,
    T duality_gap_target ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    uint outer_counter = 0;

    do {

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        y_k_old = y_k;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_temp = update_beta_ista( X, Y, y_k, L, lambda );

        uint counter = 0;

        while( ( f_beta( X, Y, y_k_temp ) > f_beta_tilda( X, Y, y_k_temp, y_k_old, L ) ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;

            DEBUG_PRINT( "L: " << L );
            y_k_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_fista( X, Y, Beta, L, lambda );

    } while ( ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target ));

    return Beta;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS<T>::ISTA_OPT (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T L_0, \
    T lambda,
    T duality_gap_target ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    do {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

        T f_beta = ( X*Beta_temp - Y ).squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > f_part = X*Beta - Y;
        T taylor_term_0 = f_part.squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta_temp - Beta );

        T taylor_term_1 = f_grad.transpose()*beta_diff;

        T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

        T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        while( f_beta > f_beta_tilde ) {

            L*= eta;

            Beta_temp = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

            f_beta = ( X*Beta_temp - Y ).squaredNorm();;

            beta_diff = ( Beta_temp - Beta );
            taylor_term_1 = f_grad.transpose()*beta_diff;
            taylor_term_2 = L/2.0*beta_diff.squaredNorm();

            f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        }

        Beta = ( Beta - (1.0/L)*f_grad ).unaryExpr( SoftThres<T>( lambda/L ) );

    } while ( ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target ) );

    hot_start_L = L;

    return Beta;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS<T>::ISTA (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T L_0, \
    T lambda,
    T duality_gap_target ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    uint outer_counter = 0;

    do {

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        uint counter = 0;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( f_beta( X, Y, Beta_temp ) > f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) {
            counter++;
            //DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_ista( X, Y, Beta, L, lambda );

        DEBUG_PRINT( "Duality Gap:" << duality_gap( X, Y, Beta, lambda ) );

    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    hot_start_L = L;

    return Beta;
}

//template < typename T >
//Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS< T >::update_beta_cd (
//    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
//    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
//    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {


//    Eigen::Matrix< T, Eigen::Dynamic, 1 > theta = Beta;

//    for( int i = 0; i < p ; i++ ) {

//        Eigen::Matrix< T, Eigen::Dynamic, 1 > A_i = X.col( i );

//        T threshold = lambda / A_i.squaredNorm();
//        T norm_factor = static_cast<T>( 1 )/( A_i.transpose()*A_i );

//        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > A_negative_i ( n, p - 1 );

//        if( i > 0 ) {

//            std::cout << X.block( 0, 0, n, i ).rows() << "x" << X.block( 0, 0, n, i ).cols() << std::endl;
//            std::cout << X.block( 0, i + 1, n, p - i - 1 ).rows() << "x" << X.block( 0, i + 1, n, p - i - 1 ).cols() << std::endl;

//            A_negative_i << X.block( 0, 0, n, i ), X.block( 0, i + 1, n, p - i - 1 );
//        } else {
////                std::cout << A_negative_i.rows() << "x" << A_negative_i.cols() << std::endl;
////                std::cout << X.block( 0, 1, n, p - 1 ).rows() << "x" << X.block( 0, 1, n, p - 1 ).cols() << std::endl;
//            A_negative_i << X.block( 0, 1, n, p - 1 );
//        }

//        Eigen::Matrix< T, Eigen::Dynamic, 1 > x_negative_i( p - 1 );

//        if( i > 0 ) {
//            std::cout << theta.rows() << " x " <<  theta.cols() << std::endl;
//            std::cout << x_negative_i.rows() << " x " <<  x_negative_i.cols() << std::endl;

//            theta.block( 0, 0, i, 1 );
//            theta.block( i + 1, 0, p - i - 1, 1 );

////                x_negative_i << Beta.block( 0, 0, i, 1 ), Beta.block( i + 1, 0, p - i - 1, 1 );

//            x_negative_i << Beta.head( i ), Beta.segment( i + 1,  p - i - 1 );
//        } else {
//            x_negative_i << Beta.tail( p - 1 );
//        }

//        theta = ( norm_factor*A_i.transpose()*( Y - A_negative_i*x_negative_i ) ).unaryExpr( SoftThres<T>( threshold ) );

//    }

//    return theta;

//}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS< T >::CoordinateDescent (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
    T lambda,
    T duality_gap_target ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

    do {

        for( int i = 0; i < p ; i++ ) {

            Eigen::Matrix< T, Eigen::Dynamic, 1 > A_i = X.col( i );

            T threshold = lambda / A_i.squaredNorm();
            T norm_factor = static_cast<T>( 1 )/( A_i.transpose()*A_i );

            Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > A_negative_i ( n, p - 1 );

            if( i > 0 ) {

//                std::cout << X.block( 0, 0, n, i ).rows() << "x" << X.block( 0, 0, n, i ).cols() << std::endl;
//                std::cout << X.block( 0, i + 1, n, p - i - 1 ).rows() << "x" << X.block( 0, i + 1, n, p - i - 1 ).cols() << std::endl;

                A_negative_i << X.block( 0, 0, n, i ), X.block( 0, i + 1, n, p - i - 1 );
            } else {
//                std::cout << A_negative_i.rows() << "x" << A_negative_i.cols() << std::endl;
//                std::cout << X.block( 0, 1, n, p - 1 ).rows() << "x" << X.block( 0, 1, n, p - 1 ).cols() << std::endl;
                A_negative_i << X.block( 0, 1, n, p - 1 );
            }

            Eigen::Matrix< T, Eigen::Dynamic, 1 > x_negative_i( p - 1 );

            if( i > 0 ) {

//                Beta.block( 0, 0, i, 1 );
//                Beta.block( i + 1, 0, p - i - 1, 1 );

//                x_negative_i << Beta.block( 0, 0, i, 1 ), Beta.block( i + 1, 0, p - i - 1, 1 );

                x_negative_i << Beta.head( i ), Beta.segment( i + 1,  p - i - 1 );
            } else {
                x_negative_i << Beta.tail( p - 1 );
            }

//            Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
//            Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

//            return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

//            std::cout << "Beta_i before update: " << Beta( i ) << std::endl;
            Beta( i ) = soft_threshold<T>( norm_factor*A_i.transpose()*( Y - A_negative_i*x_negative_i ), threshold );
//            std::cout << "Parameter w/o thresholding: " << norm_factor*A_i.transpose()*( Y - A_negative_i*x_negative_i ) << " Lambda " << threshold << std::endl;
//            std::cout << "Beta_i after update: " << Beta( i ) << std::endl;

        }

        std::cout << "Current Duality Gap: " << duality_gap( X, Y, Beta, lambda ) << std::endl;
        std::cout << "Norm Squared of updated Beta: " << Beta.squaredNorm() << std::endl;
    } while ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target );

    return Beta;
}

template < typename T >
void X_FOS< T >::operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >&x,
                             const Eigen::Matrix< T, Eigen::Dynamic, 1 >&y ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > old_Betas;

    bool statsCont = true;

    Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic > X = Normalize( x );
    Eigen::Matrix< T , Eigen::Dynamic, 1 > Y = Normalize( y );

    n = X.rows();
    p = X.cols();

    lambda_grid = GenerateLambdaGrid( X, Y, M );

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lambda_grid.at( statsIt - 1 );

        T gap = duality_gap( X, Y, old_Betas, rStatsIt );
//        T gap = primal_objective( X, Y, beta_k, rStatsIt ) + dual_objective( X, Y, beta_k, rStatsIt );

        T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

        if( gap <= gap_target ) {

            Betas.col( statsIt - 1 ) = old_Betas;

        } else {

            DEBUG_PRINT( "Current Lambda: " << rStatsIt );
//            Betas.col( statsIt - 1 ) = ISTA_OPT( X, Y, old_Betas, 0.1, rStatsIt, gap_target );
            Betas.col( statsIt - 1 ) = CoordinateDescent( X, Y, old_Betas, rStatsIt, gap_target );

        }

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lambda_grid, Betas );
    }

    fos_fit = Betas.col( statsIt - 2 );
    lambda = lambda_grid.at( statsIt - 2 );
    optim_index = statsIt;

}

}

}

#endif // X_FOS_H
