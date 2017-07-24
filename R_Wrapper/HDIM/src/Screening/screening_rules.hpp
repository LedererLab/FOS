#ifndef SCREENING_RULES_HPP
#define SCREENING_RULES_HPP

// C System-Headers
//
// C++ System headers
#include <vector>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// Project Specific Headers
#include "../Generic/generics.hpp"

namespace hdim {

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > DualPoint(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    const T lambda ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > residual = Y - X*Beta;
    T alpha = 1.0 / ( X.transpose()*residual ).template lpNorm< Eigen::Infinity >();
    T alpha_tilde = static_cast<T>( Y.transpose()*residual ) / ( lambda * residual.squaredNorm() );

    T s = std::min( std::max( alpha_tilde, - alpha ), alpha );

    return s*residual;
}

template < typename T >
T DualityGap2 ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Nu,
                const T lambda ) {

    T f_beta = 0.5*( Y - X*Beta ).squaredNorm() + lambda*Beta.template lpNorm< 1 >();
    T d_nu = 0.5*Y.squaredNorm() - square( lambda ) / 2.0 *( Nu - Y * 1.0/lambda ).squaredNorm();

    return f_beta - d_nu;

}

template < typename T >
std::vector< unsigned int > SafeActiveSet (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& center,
    const T radius ) {

    unsigned int p = X.cols();
    std::vector< unsigned int > active_indices;

    for( unsigned int j = 0; j < p ; j ++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > X_j = X.col( j );

        T safe_region = std::abs( static_cast<T>( X_j.transpose() * center ) ) + radius * X_j.norm();

        if( safe_region >= static_cast<T>( 1 ) ) {
            active_indices.push_back( j );
        }

    }

    return active_indices;
}

}


#endif // SCREENING_RULES_HPP
