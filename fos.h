#ifndef FOS_H
#define FOS_H

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
#include "fosalgorithm.h"
#include "fos_debug.h"
#include "fos_generics.h"

template < typename T >
/*!
 * \brief The main FOS algorithim
 */
class FOS {

  public:
    FOS( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x, Eigen::Matrix< T, Eigen::Dynamic, 1 > y );
    void Algorithm();

  private:
    Eigen::Matrix< T, 1, Eigen::Dynamic > GenerateRS();
    bool ComputeStatsCond(uint stats_it, uint r_stats_it, const Eigen::Matrix<T, 1, Eigen::Dynamic> &rs );
    T DualityGap( uint r_stats_it );
    T DualityGapTarget( uint r_stats_it );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y;

    const T C = 0.75;
    const uint M = 100;
    const T gamma = 1.0;

    T rMax;
    T rMin;

    bool statsCont = true;
    uint statsIt = 1;

    uint loop_index = 0;

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
FOS< T >::FOS(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x, Eigen::Matrix<T, Eigen::Dynamic, 1 > y ) : X( x ), Y( y ) {

}

//Free functions

template < typename T >
T compute_lp_norm( T& matrix, int norm_type ) {
    return matrix.template lpNorm< norm_type >();
}

template < typename T >
T compute_sqr_norm( T& matrix ) {
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
    T sqr_part = static_cast<T>( val );
    return sqr_part * sqr_part;
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
T abs_max( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > matrix ) {
    return static_cast<T>( matrix.cwiseAbs().maxCoeff() );
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

        log_space_vector( 0, i ) = static_cast<T>( pow( 10.0, lin_step ) );
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
 * \param rs
 * \return True if outer loop should continue, false otherwise
 */
bool FOS<T>::ComputeStatsCond( uint stats_it, uint r_stats_it, const Eigen::Matrix< T, 1, Eigen::Dynamic >& rs ) {

    bool stats_cond = true;

    for ( uint i = 1; i < stats_it; i++ ) {

        auto rk = rs( 0, i );

        T abs_max_betas = abs_max<T>( Betas.col( statsIt ) - Betas.col( i ) );
        T check_cond = static_cast<T>( X.rows() ) / ( static_cast<T>( r_stats_it ) + static_cast<T>( rk ) ) * abs_max_betas;

        stats_cond &= check_cond <= C;
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
Eigen::Matrix< T, 1, Eigen::Dynamic > FOS<T>::GenerateRS() {

    auto cross_prod = X.transpose() * Y;
    DEBUG_PRINT( "Cross Product X and Y: \n" << cross_prod );

    T rMax = 2.0*cross_prod.template lpNorm< Eigen::Infinity >();
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
T FOS<T>::DualityGapTarget( uint r_stats_it ) {
    T r_stats_it_f = static_cast<T>( r_stats_it );
    return square(C) * square( r_stats_it_f ) / static_cast<T>( X.rows() );
}

template < typename T >
/*!
 * \brief Compute the duality gap
 * \param r_stats_it
 * \return
 */
T FOS<T>::DualityGap( uint r_stats_it ) {

    T rStatsIt_f = static_cast<T>( r_stats_it );

    //Duality Gap Set-Up
    auto beta_t = Betas.col( statsIt - 1 );

    auto x_beta_t_prod = X * beta_t;
    auto error = x_beta_t_prod - Y;

    auto x_cross_error = X.transpose() * error;

    auto twice_x_cross_error = 2.0 * x_cross_error;

    auto alternative = rStatsIt_f/( twice_x_cross_error.template lpNorm< Eigen::Infinity >() );
    T alt = static_cast<T>( alternative );

    T y_cross_error = static_cast<T>( Y.transpose() * error );
    auto negative_error = Y - x_beta_t_prod;

    auto alternative_2 = ( -1.0 / negative_error.squaredNorm() ) * y_cross_error;

    T alt_2 = static_cast<T>( alternative_2 );

    auto s = std::min( std::max( alt, alt_2 ), -1.0*alt );
    auto nu_t = -1.0 * ( 2 * s / rStatsIt_f ) * error;

    //Compute Duality Gap
    auto f_beta = error.squaredNorm() + rStatsIt_f * beta_t.template lpNorm < 1 >() ;
    DEBUG_PRINT( "Primal Objective Function ( F Beta ): " << f_beta );

    auto ret_val = nu_t + ( 2.0 / rStatsIt_f ) * Y;
    auto d_nu = -0.25* square( rStatsIt_f ) * ret_val.squaredNorm() - Y.squaredNorm();
    DEBUG_PRINT( "Dual Function (D Nu): " << d_nu );

    auto duality_gap = static_cast<T>( f_beta ) - static_cast<T>( d_nu );
    DEBUG_PRINT( "Duality Gap = " << duality_gap );


    return duality_gap;

}

template< typename T >
/*!
 * \brief Run the main FOS algorithm
 *
 * Calling this function will run the FOS algorithm using the values of
 * X and Y that were instantiated with the class constructor.
 *
 */
void FOS< T >::Algorithm() {

    Normalize( X );
    Normalize( Y );

    auto rs = GenerateRS();

    bool statsCont = true;
    uint statsIt = 1;
    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;
        old_Betas = Betas.col( statsIt - 2 );
        auto rStatsIt = rs( 0, statsIt - 1 );

        //Inner Loop
        do {

            loop_index ++;
            DEBUG_PRINT( "Inner loop #: " << loop_index );

            T duality_gap = DualityGap( rStatsIt );
            T duality_gap_target = DualityGapTarget( rStatsIt );

            //Criteria meet, exit loop
            if( duality_gap <= duality_gap_target ) {

                DEBUG_PRINT( "Duality gap is below specified threshold, exiting inner loop." );
                Betas.col( statsIt - 1 ) = old_Betas;
                break;

            } else {

                DEBUG_PRINT( "Duality gap is " << duality_gap << " gap target is " << duality_gap_target );

                T rStatsIt_f = static_cast<T>( rStatsIt );
                DEBUG_PRINT( "Current Lambda: " << rStatsIt_f );
                old_Betas = Betas.col( statsIt - 1 ) = FistaFlat<T>( Y, X, old_Betas, 0.5*rStatsIt_f );
                DEBUG_PRINT( "L2 Norm of Updated Betas: " << old_Betas.squaredNorm() );

            }

        } while ( true );

        statsCont = ComputeStatsCond( statsIt, rStatsIt, rs );

        //auto avfosfit = Betas.col( statsIt );
    }

}


#endif // FOS_H
