#ifndef ISTA_H
#define ISTA_H

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
#include "../Generic/algorithm.h"
#include "../Generic/debug.h"
#include "../Generic/generics.h"

//template <typename T> int sgn(T val) {
//    return (T(0) < val) - (val < T(0));
//}
template <typename T>
T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template < typename T >
T pos_part( T x ) {
    return std::max( x, static_cast<T>(0.0) );
}

template < typename T >
T soft_threshold( T x, T y ) {
    T sgn_T = static_cast<T>( sgn(x) );
    return sgn_T*pos_part( std::abs(x) - y );
}

struct SoftThres_f {

    SoftThres_f( float lambda_in ) : lambda( lambda_in ) {}

    typedef float result_type;
    float operator()( float x ) const {
        return soft_threshold<float>( x, lambda );
    }

  private:
    float lambda;
};

struct SoftThres_d {

    SoftThres_d( double lambda_in ) : lambda( lambda_in ) {}

    typedef double result_type;
    double operator()( double x ) const {
        return soft_threshold<double>( x, lambda );
    }

  private:
    double lambda;
};

template < typename T >
struct SoftThres {

    SoftThres( T lambda_in ) : lambda( lambda_in ) {}

    typedef T result_type;
    T operator()( T x ) const {
        return soft_threshold<T>( x, lambda );
    }

  private:
    T lambda;
};

template < typename T >
inline T prox( T x, T lambda ) {
    return ( std::abs(x) >= lambda )?( x - sgn( x )*lambda ):( 0 );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > soft_threshold_mat(
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat,
    const T lambda ) {

    Eigen::Matrix<T, Eigen::Dynamic, 1 > mat_x( mat.rows() );

//    #pragma omp parallel for collapse(2)
    for( uint i = 0; i < mat.rows() ; i ++ ) {
        mat_x( i ) =  prox( mat( i ), lambda );
    }

    return mat_x;
}

template < typename T >
T f_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_beta = X*Beta - Y;
    return f_beta.squaredNorm();
//    return( compute_sqr_norm( X*Beta - Y ) );

}

template < typename T >
T f_beta_tilda (
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
Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0f*( X.transpose()*( X*Beta - Y ) );

//    return soft_threshold_mat<T>( Beta - (1.0/L)*f_grad, thres/L );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0f/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );
//    return soft_threshold_mat<T>( beta_to_modify, thres/L );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, \
    uint num_iterations, \
    T L_0, \
    T lambda ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

//    #pragma omp parallel for
    for( uint i = 0; i < num_iterations; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k_less_1 = Beta;

        uint counter = 0;

        Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( ( f_beta( X, Y, Beta ) > f_beta_tilda( X, Y, Beta, Beta_k_less_1, L ) ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );

            L*= eta;
            Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        }
    }

    return Beta;

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
T square( const T& val ) {
    return val * val;
}

template < typename T >
T duality_gap_target( T gamma, T C, T r_stats_it, uint n ) {

    T n_f = static_cast<T>( n );
    return gamma*square( C )*square( r_stats_it )/n_f;

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
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_ISTA (
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

    T delta_k, delta_k_less_1 = 0;

    do {

        delta_k = duality_gap( X, Y, Beta, lambda );

        outer_counter ++;
        DEBUG_PRINT( "Outer loop iteration: " << outer_counter );

        std::cout << "Outer loop iteration: " << outer_counter << std::endl;
        std::cout << "Duality gap is " << delta_k << " gap target is " << duality_gap_target << std::endl;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_k_less_1 = Beta;

        uint counter = 0;

        Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( ( f_beta( X, Y, Beta ) > f_beta_tilda( X, Y, Beta, Beta_k_less_1, L ) ) ) {

            counter++;
            DEBUG_PRINT( "Backtrace iteration: " << counter );
//            std::cout << "Backtrace iteration: " << counter << std::endl;

            L*= eta;
            Beta = update_beta( X, Y, Beta_k_less_1, L, lambda );

        }

        delta_k_less_1 = duality_gap( X, Y, Beta, lambda );

//        std::cout << "Squared norm of Beta tilde " << Beta.squaredNorm() << std::endl;

    } while ( ( duality_gap( X, Y, Beta, lambda ) > duality_gap_target ) );

    return Beta;

}


#endif // ISTA_H
