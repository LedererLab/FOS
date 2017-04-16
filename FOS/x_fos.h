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
    X_FOS( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x, const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y );

    /*!
     * \brief Run the main X_FOS algorithm
     *
     * Calling this function will run the X_FOS algorithm using the values of
     * X and Y that were instantiated with the class constructor.
     *
     */
    void Run();

    T ReturnLambda();
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
    uint ReturnOptimIndex();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnSupport();

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > fos_fit;
    T lambda;
    uint optim_index;

  private:
    Eigen::Matrix< T, 1, Eigen::Dynamic > GenerateLambdaGrid (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        uint M );

    bool ComputeStatsCond(
        T C,
        uint stats_it,
        T r_stats_it,
        const Eigen::Matrix< T, 1, Eigen::Dynamic >& lambdas,
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas );

    T DualityGapTarget( uint r_stats_it );

    T duality_gap_target( T gamma, T C, T r_stats_it, uint n );

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

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_ista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_fista (const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
            T L,
            T thres );

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

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > y_k_old;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x_k_less_1;

    const T C = 0.75;
    const uint M = 100;
    const T gamma = 1;

    T rMax;
    T rMin;

    bool statsCont = true;

    uint loop_index = 0;

    uint statsIt = 1;
    Eigen::Matrix< T, 1, Eigen::Dynamic > lambda_grid;

    T t_k = 1;
    T hot_start_L = 0.1;
    const T L_0 = 0.1;

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
X_FOS< T >::X_FOS( const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x,
                   const Eigen::Matrix<T, Eigen::Dynamic, 1 >& y ) : X( x ), Y( y ) {

    x_k_less_1 = Eigen::Matrix<T, Eigen::Dynamic, 1 >::Zero( X.rows(), 1 );

}

//template< typename T >
//X_FOS<T>::X_FOS & operator=( const X_FOS &rhs ) {

//}

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
Eigen::Matrix<T, Eigen::Dynamic, 1> X_FOS<T>::ReturnSupport() {

    T C_t = static_cast<T>( C );
    T n_t = static_cast<T>( X.rows() );

    return fos_fit.unaryExpr( SupportSift<T>( C_t, lambda, n_t ) );

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
                                 const Eigen::Matrix< T, 1, Eigen::Dynamic >& lambdas,
                                 const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                 const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas ) {

    bool stats_cond = true;

    for ( uint i = 1; i <= stats_it; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_k = Betas.col( i - 1 );
        T rk = lambdas( 0, i - 1 );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = Betas.col( stats_it - 1 );
        beta_diff -= beta_k;
        T abs_max_betas = beta_diff.cwiseAbs().maxCoeff();

        T n = static_cast<T>( X.rows() );

        T check_parameter = n*abs_max_betas / ( r_stats_it + rk );

        DEBUG_PRINT( "Check Parameter: " << check_parameter );

//        if( !( check_parameter <= C ) ) {
//            stats_cond = false;
//            break;
//        }

        stats_cond &= ( check_parameter <= C );
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
Eigen::Matrix< T, 1, Eigen::Dynamic > X_FOS<T>::GenerateLambdaGrid (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    uint M ) {

    T rMax = 2.0*L_infinity_norm( X.transpose() * Y );
    T rMin = 0.001*rMax;

    DEBUG_PRINT( "rMax " << rMax << " rMin " << rMin );

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
    return square(C) * square( r_stats_it_f ) / static_cast<T>( X.rows() );
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

    T alternative = r_stats_it /( L_infinity_norm( 2.0f*X.transpose()*error ) );

    T alt_part_1 = static_cast<T>( Y.transpose()*error );

    T alternative_0 = alt_part_1/( error_sqr_norm );

    T s = std::min( std::max( alternative, alternative_0 ), -1.0f*alternative );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_part = ( - 2.0f*s / r_stats_it ) * error + 2.0f/r_stats_it*Y;

    T d_nu = 0.25*square( r_stats_it )*nu_part.squaredNorm() - Y.squaredNorm();

    return f_beta + d_nu;
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

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_beta = X*Beta - Y;
    return f_beta.squaredNorm();
//    return( compute_sqr_norm( X*Beta - Y ) );

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
            y_k_temp = update_beta_ista( X, Y, y_k_old, L, lambda );

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

    uint outer_counter = 0;

    do {

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        uint counter = 0;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > to_modify = Beta - (1.0/L)*f_grad;
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        T f_beta = ( X*Beta_temp - Y ).squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > f_part = X*Beta - Y;
        T taylor_term_0 = f_part.squaredNorm();

        Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta_temp - Beta );

        T taylor_term_1 = f_grad.transpose()*beta_diff;

        T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

        T f_beta_tilde = taylor_term_0 + taylor_term_1 + taylor_term_2;

        while( f_beta > f_beta_tilde ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;

            DEBUG_PRINT( "L: " << L );

            to_modify = Beta - (1.0/L)*f_grad;
            Beta_temp = to_modify.unaryExpr( SoftThres<T>( lambda/L ) );

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

        while( ( f_beta( X, Y, Beta_temp ) > f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {
            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_ista( X, Y, Beta, L, lambda );

    } while ( ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target ) );

    hot_start_L = L;

    return Beta;
}

template< typename T >
void X_FOS< T >::Run() {

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    bool statsCont = true;
//    uint statsIt = 1;

    X = Normalize( X );
    Y = Normalize( Y );

    lambda_grid = GenerateLambdaGrid( X, Y, M );
    DEBUG_PRINT( "Lambda grid: " << lambda_grid );

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lambda_grid( statsIt - 1 );

        Eigen::Matrix< T , Eigen::Dynamic, 1 > beta_k = Betas.col( statsIt - 1 );

        T gap = duality_gap( X, Y, beta_k, rStatsIt );

        uint n = static_cast< uint >( X.rows() );
        T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

        if( gap <= gap_target ) {

            Betas.col( statsIt - 1 ) = old_Betas;

        } else {

            DEBUG_PRINT( "Current Lambda: " << rStatsIt );

            Betas.col( statsIt - 1 ) = ISTA( X, Y, old_Betas, hot_start_L, rStatsIt, gap_target );
//            Betas.col( statsIt - 1 ) = X_ISTA( X, Y, old_Betas, hot_start_L, rStatsIt, gap_target );
            old_Betas = Betas.col( statsIt - 1 );

            DEBUG_PRINT( "L2 Norm of Betas: " << Betas.squaredNorm() );

        }

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lambda_grid, X, Betas );
    }

    fos_fit = Betas.col( statsIt - 2 );
    lambda = lambda_grid( statsIt - 1 );
    optim_index = statsIt;

    DEBUG_PRINT( fos_fit );
    DEBUG_PRINT( optim_index );

}

}

}

#endif // X_FOS_H
