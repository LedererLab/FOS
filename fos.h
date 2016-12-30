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
//#include <spams.h> // _fistaFlat
// Project Specific Headers
#include "fosalgorithm.h"
#include "fos_typetraits.h"

template < typename T >
class FOS {

  public:
    FOS( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x, Eigen::Matrix< T, Eigen::Dynamic, 1 > y );
    void Algorithm();
    void Demo(  T lower_bound, T upper_bound, uint num_elements );

  private:
    T Mean(  Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat );
    T StdDev(  Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat );
    void Normalize(  Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat );
    void Normalize(  Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat );
    Eigen::Matrix< T, 1, Eigen::Dynamic > LogScaleVector( T lower_bound, T upper_bound, uint num_elements );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y;

    T C = 0.75;
    uint M = 100;
    T rMax;
    T rMin;
    float statsIt = 1;

    uint loop_index = 0;

};

template< typename T >
FOS< T >::FOS(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x, Eigen::Matrix<T, Eigen::Dynamic, 1 > y ) : X( x ), Y( y ) {

}

template< typename T >
T FOS< T >::StdDev( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat ) {

    Eigen::RowVectorXf mean = mat.colwise().mean();
    return ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();
}


template< typename T >
void FOS< T >::Normalize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
Eigen::Matrix< T, 1, Eigen::Dynamic > FOS< T >::LogScaleVector( T lower_bound, T upper_bound, uint num_elements ) {

    T min_elem = static_cast<T>( log10(lower_bound) );
    T max_elem = static_cast<T>( log10(upper_bound) );
    T delta = max_elem - min_elem;

    Eigen::Matrix< T, 1, Eigen::Dynamic > log_space_vector;
    log_space_vector.resize( num_elements );

    for ( uint i = 0; i < num_elements ; i ++ ) {

        T step = (T)i/(T)( num_elements - 1 );
        auto lin_step = delta*step + min_elem;

        log_space_vector( 0, i ) = static_cast<T>( pow( 10.0, lin_step ) );
    }

    return log_space_vector;
}

template< typename T >
void FOS< T >::Normalize( Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}

//template < typename T >
//void FOS< T >::SetUp() {
//    //
//}

//template < typename T >
//void FOS< T >::DualityGapPrequel() {
//    //
//}

//template < typename T >
//void FOS< T >::DualityGap() {
//    //
//}

//template < typename T >
//void FOS< T >::UpdateCriterion() {
//    //
//}

template < typename T >
T compute_lp_norm( T& matrix, int norm_type ) {
    return matrix.template lpNorm< norm_type >();
}

template < typename T >
T compute_sqr_norm( T& matrix ) {
    return matrix.squaredNorm();
}

template < typename T>
T square( T& val ) {
    T sqr_part = static_cast<T>( val );
    return sqr_part * sqr_part;
}

template < typename T >
T abs_max( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > matrix ) {
    return static_cast<T>( matrix.cwiseAbs().maxCoeff() );
}

template< typename T >
void FOS< T >::Demo( T lower_bound, T upper_bound, uint num_elements ) {
    std::cout << LogScaleVector( lower_bound, upper_bound, num_elements ) << std::endl;
}

template< typename T >
void FOS< T >::Algorithm() {

    Normalize( X );
    Normalize( Y );

    auto cross_prod = X.transpose() * Y;
    rMax = 2.0*cross_prod.template lpNorm< Eigen::Infinity >();
    rMin = 0.001*rMax;

    DEBUG_PRINT( "rMax " << rMax << " rMin " << rMin );

    auto rs = LogScaleVector( rMax, rMin, M );

    bool statsCont = true;
    uint statsIt = 1;
    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;
        old_Betas = Betas.col( statsIt - 2 );
        auto rStatsIt = rs( 0, statsIt - 1 );

        do {

            loop_index ++;
            DEBUG_PRINT( "Inner loop #: " << loop_index );

            T rStatsIt_f = static_cast<T>( rStatsIt );

            //Duality Gap Prequel
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

            auto ret_val_3 = nu_t + ( 2.0 / rStatsIt_f ) * Y;
            auto d_nu = -0.25* square( rStatsIt_f ) * ret_val_3.squaredNorm() - Y.squaredNorm();

            auto duality_gap = static_cast<T>( f_beta ) - static_cast<T>( d_nu );

            DEBUG_PRINT( "Duality Gap = " << duality_gap );

            auto duality_gap_target = square(C) * square( rStatsIt_f ) / static_cast<T>( X.rows() );

            //Criteria meet, exit loop
            if( duality_gap <= duality_gap_target ) {

                DEBUG_PRINT( "Duality gap is below specified threshold, exiting inner loop." );

                Betas.col( statsIt - 1 ) = old_Betas;
                break;

            } else {

                DEBUG_PRINT( "Duality gap is " << duality_gap << " gap target is " << duality_gap_target );

                old_Betas = Betas.col( statsIt - 1 ) = FistaFlat<T>( Y, X, old_Betas, 0.5*rStatsIt_f );
//                DEBUG_PRINT( "Return value from fistaFlat" << std::endl << old_Betas << std::endl );

                DEBUG_PRINT( "Beta Matrix:" << std::endl << Betas << std::endl );
//                old_Betas = Betas.col( statsIt - 1 );
            }

        } while ( true );

        for ( uint i = 1; i < statsIt; i++ ) {

            auto rk = rs( 0, i );

            T abs_max_betas = abs_max<T>( Betas.col( statsIt ) - Betas.col( i ) );
            T check_cond = static_cast<T>( X.rows() ) / ( static_cast<T>( rStatsIt ) + static_cast<T>( rk ) ) * abs_max_betas;

            statsCont &= check_cond <= C;
        }

        auto avfosfit = Betas.col( statsIt );
    }

}


#endif // FOS_H
