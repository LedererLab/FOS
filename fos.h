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

template < typename T >
class FOS {

  public:
    FOS( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x, Eigen::Matrix< T, Eigen::Dynamic, 1 > y );
    void Algorithm();

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

        log_space_vector( 0, i ) = (T)pow( 10.0, lin_step );
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
T ComputeNorm( T& matrix, int norm_type ) {
    return matrix.template lpNorm< norm_type >();
}

template< typename T >
void FOS< T >::Algorithm() {

    Normalize( X );
    Normalize( Y );

    auto cross_prod = X.transpose() * Y;
    auto rMax = 2*cross_prod.template lpNorm< Eigen::Infinity >();

    auto rs = LogScaleVector( rMin, rMax, M );

    bool statsCont = true;
    uint statsIt = 1;
    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.rows(), X.cols() );
    auto r_tilda = rs.col( M - 1 );

    //Outer Loop
    while( statsCont && statsIt < M ) {

        statsIt ++;
        old_Betas = Betas.col( statsIt - 1 );
        auto rStatsIt = rs.col( statsIt );

        do {
            //Duality Gap Prequel
            auto beta_t = Betas.col( statsIt );
            std::cout << beta_t.rows() << "x" << beta_t.cols() << std::endl;
            auto error = X*beta_t- Y;

            auto ret_val = 2.0 * X.transpose() * error;

            auto alternative = rStatsIt / ret_val.template lpNorm< Eigen::Infinity >();

            auto ret_val_2 = Y - X * beta_t;
            auto alternative_2 = -1.0 / ret_val_2.template lpNorm< 2 >() * Y.transpose() * error;

            auto s = std::max( { alternative, alternative_2, -1*alternative} );
            auto nu_t = -1.0 * ( 2 * s / rStatsIt ) * error;

            //Compute Duality Gap
            auto f_beta = pow( ret_val_2.squaredNorm() + rStatsIt*beta_t.template lpNorm < 1 >() );
            auto ret_val_3 = 0.25* pow( rStatsIt, 2.0 ) * nu_t + ( 2.0 / rStatsIt ) * Y;

            auto d_nu = ret_val_3 - Y.squaredNorm();
            auto duality_gap = f_beta - d_nu;

        } while ( 0 );

    }

}


#endif // FOS_H
