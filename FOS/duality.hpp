#ifndef DUALITY_H
#define DUALITY_H

// C System-Headers
//
// C++ System headers
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <memory>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// Project Specific Headers
#include "../Generic/generics.hpp"

template < typename T >
class Duality {

    void operator()(const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
                    T r_stats_it );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > NuTilde();
    T DualityGap();

  private:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_tilde;
    T duality_gap;
    bool ready = false;
};

template < typename T >
void Duality< T > ::operator()(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
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

    nu_tilde = s*error;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_part = ( - 2.0*s / r_stats_it ) * error + 2.0/r_stats_it*Y;

    T d_nu = 0.25*square( r_stats_it )*nu_part.squaredNorm() - Y.squaredNorm();

    duality_gap = f_beta + d_nu;

    ready = true;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > Duality < T >::NuTilde() {

    if( !ready ) {
        throw std::runtime_error( "Data is not initialized, the () operator must be used before data is returned." );
    }

    return nu_tilde;
}

template < typename T >
T Duality < T >::NuTilde() {

    if( !ready ) {
        throw std::runtime_error( "Data is not initialized, the () operator must be used before data is returned." );
    }

    return duality_gap;
}

#endif // DUALITY_H
