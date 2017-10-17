#ifndef FOS_GENERICS_H
#define FOS_GENERICS_H

// C System-Headers
#include <fenv.h>

#ifdef __linux__
#include <tgmath.h>
#elif _WIN32
#include <ctgmath>
#endif

// C++ System headers
#include <chrono>
#include <fstream>      // std::ifstream
#include <vector>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// OpenMP
//
// Project Specific Headers
//

/*! \file
 *  \brief Generic linear algebra functions.
 */

namespace hdim {

template < typename T >
T StdDev( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& vect ) {
    if( vect.size() == 0 ) {
        return static_cast<T>( 0 );
    }

    T K = vect( 0 );
    T n = static_cast<T>(0.0);
    T sum = static_cast<T>(0.0);
    T sum_sqr = static_cast<T>(0.0);
    T variance = static_cast<T>(0.0);

    for( unsigned int i = 0; i < vect.size() ; i ++ ) {

        n = n + static_cast<T>(1.0);

        T x = vect( i );

        sum += x - K;
        sum_sqr += ( x - K ) * ( x - K );
        variance = ( sum_sqr - (sum*sum)/n)/( n - static_cast<T>(1.0) );

    }

    return std::sqrt( variance );
}

template < typename T >
/*!
 * \brief Set the mean of a matrix to 0 and the standard deviation to 1.
 *
 * Note this function is done in place, that is the input matrix is modified.
 *
 * \param mat
 *
 * An n x m matrix to be normalized.
 */
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Normalize(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template< typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Normalize(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
/*!
 * \brief Overloaded version of hdim::Normalize to
 * accomadate vectors
 */
Eigen::Matrix<T, Eigen::Dynamic, 1 > Normalize(
    const Eigen::Matrix<T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
/*!
 * \brief Overloaded version of hdim::Normalize to
 * accomadate vectors
 */
Eigen::Matrix<T, Eigen::Dynamic, 1 > Normalize(
    Eigen::Matrix<T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template< typename T >
/*!
 * \brief Generate a matrix using a function that depends on
 * row and column indices.
 *
 * That is \f$ \forall i, j A_{i,j} = f( i, j ) \f$
 * for some function f and an input matrix A.
 *
 * \param num_rows
 * Number of rows the output matrix should have
 *
 * \param num_cols
 * Number of columns the output matrix should have
 *
 * \return
 * A matrix populated with values assigned by mat_func( i, j )
 */
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> build_matrix( unsigned int num_rows, unsigned int num_cols, T (*mat_func)(unsigned int,unsigned int) ) {

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat( num_rows, num_cols );

    for( unsigned int i = 0; i < num_rows ; i ++ ) {

        for( unsigned int j = 0; j < num_cols ; j++ ) {
            mat( i, j ) = (*mat_func)( i, j );
        }
    }

    return mat;
}

template< typename T >
/*!
 * \brief sweep_matrix
 * \param mat
 */
void sweep_matrix( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, T (*mat_func)(unsigned int,unsigned int) ) {

    for( unsigned int i = 0; i < mat.rows() ; i ++ ) {

        for( unsigned int j = 0; j < mat.cols() ; j++ ) {
            mat( i, j ) = (*mat_func)( i, j );
        }
    }
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
std::vector < T > LogScaleVector( T lower_bound, T upper_bound, unsigned int num_elements ) {

    T min_elem = static_cast<T>( log10(lower_bound) );
    T max_elem = static_cast<T>( log10(upper_bound) );
    T delta = max_elem - min_elem;

    std::vector < T > log_space_vector;
    log_space_vector.reserve( num_elements );

    for ( unsigned int i = 0; i < num_elements ; i ++ ) {

        T step = static_cast<T>( i )/static_cast<T>( num_elements - 1 );
        auto lin_step = delta*step + min_elem;

        log_space_vector.push_back( static_cast<T>( std::pow( 10.0, lin_step ) ) );
    }

    return log_space_vector;
}

template < typename T >
/*!
 * \brief Functor to convert vector of values into support vector
 *
 * Designed to be applied with Eigen::Matrix::unaryExpr or the like.
 */
struct Binarize {

    typedef T result_type;
    T operator()( T x ) const {
        return static_cast<T>( x == static_cast<T>( 0 ) );
    }

};

template <typename T>
/*!
 * \brief The sign function
 * defined as
 *
 * \f[  sgn(x) \equiv
 *  \begin{cases}
 *      1 & \text{if } x > 0 \\
 *      0 & \text{if } x = 0 \\
 *     -1& \text{if } x < 0
 *  \end{cases}
 * \f]
 */
T sgn(T val) {
    return static_cast<T>( T(0) < val ) - ( val < T(0) );
}

template < typename T >
/*!
 * \brief The positive part function
 * defined as
 *
 * \f[  x^{+} \equiv
 *  \begin{cases}
 *      x & \text{if } x \geq 0\\
 *      0 & \text{if } x < 0\\
 *  \end{cases}
 * \f]
 */
T pos_part( T x ) {
    return std::max( x, static_cast<T>(0.0) );
}

template < typename T >
/*!
 * \brief The proximal ( soft thresholding ) operator defined as
 *
 * \f[  \tau( x, y ) \equiv sgn(x) \left( \lvert x \rvert  - y \right)^{+}
 * \f]
 *
 * \param val
 * \return
 */
T soft_threshold( T x, T y ) {
    T sgn_T = static_cast<T>( sgn(x) );
    return sgn_T*pos_part( std::abs(x) - y );
}

template < typename T >
/*!
 * \brief Soft Threshold functor used to apply
 * hdim::soft_threshold to each element in a matrix or vector.
 *
 * Designed to be applied with Eigen::Matrix::unaryExpr or the like.
 */
struct SoftThres {

    /*!
     * \brief Initialize proximal operator, note that the
     * term lambda_in takes the place of 'y' in the definition
     * of the proximal operator.
     *
     * \param lambda_in
     *
     * The equivalent of 'y' in the definition of the proximal operator
     * the value for 'x' will be provided by the matrix element.
     */
    SoftThres( T lambda_in ) : lambda( lambda_in ) {}

    typedef T result_type;
    T operator()( T x ) const {
        return soft_threshold<T>( x, lambda );
    }

  private:
    T lambda;
};

template < typename T >
/*!
 * \brief A functional equivalent of hdim::soft_threshold, but possibly faster.
 */
inline T prox( T x, T lambda ) {
    return ( std::abs(x) >= lambda )?( x - sgn( x )*lambda ):( 0 );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > soft_threshold_mat(
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat,
    const T lambda ) {

    Eigen::Matrix<T, Eigen::Dynamic, 1 > mat_x( mat.rows() );

//    #pragma omp parallel for collapse(2)
    for( unsigned int i = 0; i < mat.rows() ; i ++ ) {
        mat_x( i ) =  prox( mat( i ), lambda );
    }

    return mat_x;
}

template < typename T >
struct SupportSift {

    SupportSift( T C, T r_tilde, T n ) : cut_off( std::abs( static_cast<T>( 6 )*C*r_tilde/n ) ) {}

    typedef T result_type;
    T operator()( T x ) const {
        return ( std::abs(x) >= cut_off )?( static_cast<T>( 1 ) ):( static_cast<T>( 0 ) );
    }

  private:
    T cut_off;
};

template< typename T >
Eigen::Matrix< int, Eigen::Dynamic, 1 > GenerateSupport(
    const Eigen::Matrix<T, Eigen::Dynamic, 1 >& coefficients,
    T cut_off ) {

    Eigen::Matrix< int, Eigen::Dynamic, 1 > support( coefficients.rows() );

    for( unsigned int i = 0; i < coefficients.size() ; i ++ ) {
        T x = coefficients( i );
        support( i ) = ( std::abs( x ) >= cut_off );
    }

    return support;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > negative_index(
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat_in,
    int index ) {

    unsigned int n = mat_in.rows();
    unsigned int p = mat_in.cols();

    //            MatrixT A_negative_i ( n, p - 1 );

    //            if( i > 0 ) {
    //                A_negative_i << X.block( 0, 0, n, i ), X.block( 0, i + 1, n, p - i - 1 );
    //            } else {
    //                A_negative_i << X.block( 0, 1, n, p - 1 );
    //            }

    //            VectorT x_negative_i( p - 1 );

    //            if( i > 0 ) {
    //                x_negative_i << Beta.head( i ), Beta.segment( i + 1,  p - i - 1 );
    //            } else {
    //                x_negative_i << Beta.tail( p - 1 );
    //            }

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > left_half = mat_in.block( 0, 0, n, index - 1 );
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > right_half = mat_in.block( 0, index + 1, n, p );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > out ( n, p - 1 );
    out << left_half, right_half;

    return out;
}

template < typename T >
T duality_gap ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
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
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > slice (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& org,
        const std::vector< unsigned int >& indices ) {

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > sliced ( org.rows(), indices.size() );

    for( unsigned int j = 0 ; j < indices.size() ; j++ ) {
        sliced.col( j ) = org.col( indices[ j ] );
    }

    return sliced;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > slice (
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& org,
        const std::vector< unsigned int >& indices ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > sliced ( indices.size() );

    for( unsigned int j = 0 ; j < indices.size() ; j++ ) {
        sliced[ j ] = org[ indices[ j ] ];
    }

    return sliced;

}

}

#endif // FOS_GENERICS_H
