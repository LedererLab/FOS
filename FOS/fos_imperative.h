#ifndef FOS_IMPERATIVE_H
#define FOS_IMPERATIVE_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//
// Project Specific Headers
#include "../Generic/algorithm.h"
#include "../Generic/debug.h"
#include "../Generic/generics.h"
#include "../ISTA/ista.h"

namespace hdim {

namespace experimental {

template < typename T >
typename T::value_type L_infinity_norm( const T& matrix ) {
    return matrix.template lpNorm< Eigen::Infinity >();
}

template < typename T >
typename T::value_type L1_norm( const T& matrix ) {
    return matrix.template lpNorm< 1 >();
}

//template < typename T >
//T compute_lp_norm( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& matrix, int norm_type ) {
//    return matrix.template lpNorm< norm_type >();
//}

//template < typename T >
//T compute_sqr_norm( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& matrix ) {
//    return matrix.squaredNorm();
//}

template < typename T >
typename T::value_type compute_sqr_norm( const T& matrix ) {
    return matrix.squaredNorm();
}

template < typename T>
/*!
 * \brief Compute the square of a value
 * \param val
 *
 * value to square
 *
 * \return The squared quantity
 */
T square( T& val ) {
    return val * val;
}

template < typename T >
/*!
 * \brief Compute the maximum of the absolute value of an Eigen::Matrix object
 *
 * \param matrix
 *
 * Matrix to work on- note that the matrix is not modified.
 *
 * \return Coeffecient-wise maximum of the absolute value of the argument
 */
T abs_max( const T& matrix ) {
    return matrix.cwiseAbs().maxCoeff();
}

template < typename T >
/*!
 * \brief Generate a vector of logarithmically equally spaced points
 *
 * There will be num_element points, beginning at log10( lower_bound )
 *  and ending at log10( upper_bound ).
 *
 * This function is semantically equivalent to the R function 'logspace'.
 *
 * \param lower_bound
 *
 * 10^x for x = smallest element in vector
 *
 * \param upper_bound
 *
 * 10^x for x = largest element in vector
 *
 * \param num_elements
 *
 * number of elements in the generated vector
 *
 * \return
 *
 * Vector of logarithmically equally spaced points
 */
Eigen::Matrix< T, 1, Eigen::Dynamic > LogScaleVector( T lower_bound, T upper_bound, uint num_elements ) {

    T min_elem = static_cast<T>( log10(lower_bound) );
    T max_elem = static_cast<T>( log10(upper_bound) );
    T delta = max_elem - min_elem;

    Eigen::Matrix< T, 1, Eigen::Dynamic > log_space_vector;
    log_space_vector.resize( num_elements );

    for ( uint i = 0; i < num_elements ; i ++ ) {

        T step = static_cast<T>( i )/static_cast<T>( num_elements - 1 );
        auto lin_step = delta*step + min_elem;

        log_space_vector( 0, i ) = static_cast<T>( std::pow( 10.0, lin_step ) );
    }

    return log_space_vector;
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
bool ComputeStatsCond( T C,
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

        stats_cond &= ( check_parameter <= C );
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
Eigen::Matrix< T, 1, Eigen::Dynamic > GenerateLambdaGrid (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    uint M ) {

    T rMax = 2.0*L_infinity_norm( X.transpose() * Y );
    T rMin = 0.001*rMax;

    DEBUG_PRINT( "rMax " << rMax << " rMin " << rMin );

    return LogScaleVector( rMax, rMin, M );

}

template < typename T >
T DualityGapTarget( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, T C, uint r_stats_it ) {
    T r_stats_it_f = static_cast<T>( r_stats_it );
    return square(C) * square( r_stats_it_f ) / static_cast<T>( X.rows() );
}

template < typename T >
T primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                    T r_stats_it ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = X*Beta - Y;
    T f_beta = error.squaredNorm() + r_stats_it*Beta.template lpNorm < 1 >();

    return f_beta;

}

template < typename T >
T dual_objective ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                   const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                   const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                   T r_stats_it ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = X*Beta - Y;

    //Compute dual point

    T alternative = r_stats_it/compute_lp_norm( 2.0f*X.transpose()*error, Eigen::Infinity );

    T alt_part_1 = -1.0*static_cast<T>( Y.transpose()*error );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > alt_part_2 = Y - X*Beta;

    T alternative_0 = alt_part_1/( alt_part_2.squaredNorm() );

    T s = std::min( std::max( alternative, alternative_0 ), -1.0f*alternative );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_t = -1.0f*( 2.0f*s / r_stats_it ) * error;

    Eigen::Matrix< T, Eigen::Dynamic, 1 >  nu_part = nu_t + 2.0f/r_stats_it*Y;

    T d_nu = 0.25* square( r_stats_it )*nu_part.squaredNorm() - Y.squaredNorm();

    return d_nu;
}

template < typename T >
T duality_gap ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                T r_stats_it ) {

    //Computation of Primal Objective

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = X*Beta - Y;

    T f_beta = error.squaredNorm() + r_stats_it*Beta.template lpNorm < 1 >();

    //Computation of Dual Objective

    //Compute dual point

    T alternative = r_stats_it /( L_infinity_norm( 2.0f*X.transpose()*error ) );

    T alt_part_1 = -1.0*static_cast<T>( Y.transpose()*error );

    T alternative_0 = alt_part_1/( compute_sqr_norm( -1.0f*error ) );

    T s = std::min( std::max( alternative, alternative_0 ), -1.0f*alternative );

    T d_nu = 0.25*square( r_stats_it )*compute_sqr_norm( -1.0f*( 2.0f*s / r_stats_it ) * error + 2.0f/r_stats_it*Y ) - Y.squaredNorm();

    return f_beta + d_nu;
}

template < typename T >
T duality_gap_target( T gamma, T C, T r_stats_it, uint n ) {

    T n_f = static_cast<T>( n );
    return gamma*square( C )*square( r_stats_it )/n_f;

}


template< typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> FOS(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X,
    Eigen::Matrix<T, Eigen::Dynamic, 1 >& Y ) {

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    const T C = 0.75;
    const uint M = 100;
    const T gamma = 1;

    bool statsCont = true;
    uint statsIt = 1;

    uint loop_index = 0;

    Normalize( X );
    Normalize( Y );

//    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >  X = Normalize( x );
//    const Eigen::Matrix< T, Eigen::Dynamic, 1 >  Y = Normalize( y );

    Eigen::Matrix< T, 1, Eigen::Dynamic > lamda_grid = GenerateLambdaGrid( X, Y, M );
    DEBUG_PRINT( "Lambda grid: " << lamda_grid );

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lamda_grid( 0, statsIt - 1 );

        //Inner Loop
        while( true ) {

            loop_index ++;
            DEBUG_PRINT( "Inner loop #: " << loop_index );

            Eigen::Matrix< T , Eigen::Dynamic, 1  > beta_k = Betas.col( statsIt - 1 );

            T gap = duality_gap( X, Y, beta_k, rStatsIt );

            uint n = static_cast< uint >( X.rows() );
            T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

            DEBUG_PRINT( "Duality gap is " << gap << " gap target is " << gap_target );

            //Criteria meet, exit loop
            if( gap <= gap_target ) {

                DEBUG_PRINT( "Duality gap is below specified threshold, exiting inner loop." );
                Betas.col( statsIt - 1 ) = old_Betas;
                loop_index = 0;

                break;

            } else {

                DEBUG_PRINT( "Current Lambda: " << rStatsIt );

                Betas.col( statsIt - 1 ) = FistaFlat<T>( Y, X, old_Betas, 0.5*rStatsIt );
                //Betas.col( statsIt - 1 ) = ISTA<T>( X, Y, old_Betas, 1, 0.1, 0.5*rStatsIt );

                old_Betas = Betas.col( statsIt - 1 );

                DEBUG_PRINT( "L2 Norm of Betas: " << Betas.squaredNorm() );

            }

        }

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lamda_grid, X, Betas );
    }

    auto avfos_fit = Betas.col( statsIt - 2 );
    auto optim_index = statsIt;

    DEBUG_PRINT( avfos_fit );
    DEBUG_PRINT( optim_index );

    return Betas;

}

}

}

#endif // FOS_IMPERATIVE_H
