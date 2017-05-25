#ifndef FISTA_H
#define FISTA_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
// Boost Headers
//
// SPAMS Headers
//
// OpenMP Headers
#include <omp.h> //OpenMP pragmas
// Project Specific Headers
#include "../ISTA/ista.h"

namespace hdim {

template< typename T >
using MatrixT = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
using VectorT = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template < typename T >
VectorT<T> FISTA (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    uint num_iterations,
    T L_0,
    T lambda ) {

    auto solver = internal::FISTA <T>();
    return solver( X, Y, Beta_0, num_iterations, L_0, lambda );
}

namespace internal {

template < typename T >
class FISTA {

  public:
    FISTA();
    VectorT<T> operator()(
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta_0,
        uint num_iterations,
        T L_0,
        T lambda );

  private:
    VectorT<T> update_beta_fista (
        const MatrixT<T>& X,
        const VectorT<T>& Y,
        const VectorT<T>& Beta,
        T L,
        T thres );

    VectorT<T> y_k;
    VectorT<T> y_k_old;

    MatrixT<T> x_k_less_1;

    T t_k = 1;
};

template < typename T >
FISTA::FISTA() {
    static_assert(std::is_floating_point< T >::value, "FISTA can only be used with floating point types.");
}

template < typename T >
VectorT<T> FISTA<T>::update_beta_fista (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta,
    T L,
    T thres ) {

    VectorT<T> x_k = Beta;

    x_k_less_1 = x_k;
    x_k = update_beta_ista( X, Y, y_k, L, thres );

    T t_k_plus_1 = ( 1.0 + std::sqrt( 1.0 + 4.0 * square( t_k ) ) ) / 2.0;
    t_k = t_k_plus_1;

    y_k = x_k + ( t_k - 1.0 ) / ( t_k_plus_1 ) * ( x_k - x_k_less_1 );

    return x_k;
}


template < typename T >
MatrixT<T> FISTA<T>::operator() (
    const MatrixT<T>& X,
    const VectorT<T>& Y,
    const VectorT<T>& Beta_0,
    uint num_iterations,
    T L_0,
    T lambda ) {

    T eta = 1.5;
    T L = L_0;

    VectorT<T> Beta = Beta_0;
    y_k = Beta;
    t_k = 1;

    uint outer_counter = 0;

    for( uint i = 0; i < num_iterations; i++ ) {

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        y_k_old = y_k;

        VectorT<T> y_k_temp = update_beta_ista( X, Y, y_k, L, lambda );

        uint counter = 0;

        while( ( f_beta( X, Y, y_k_temp ) > f_beta_tilda( X, Y, y_k_temp, y_k_old, L ) ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;

            DEBUG_PRINT( "L: " << L );
            y_k_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_fista( X, Y, Beta, L, lambda );

    }

    return Beta;

}

}

}


#endif // FISTA_H
