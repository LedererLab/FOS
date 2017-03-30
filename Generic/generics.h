#ifndef FOS_GENERICS_H
#define FOS_GENERICS_H

// C System-Headers
#include <fenv.h>
#include <tgmath.h>
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
// SPAMS Headers
//
// Armadillo Headers
#include <armadillo>
// Project Specific Headers
//

/*! \file
 *  \brief Generic linear algebra functions.
 */

template < typename T >
/*!
 * \brief Read a .csv file into an Eigen Matrix
 *
 * Files must -not- have header information of any kind (e.g. row/col labels etc. )
 * Rows are determined by line breakers, columns are determined by comma-delimiter.
 *
 * \param file_path
 *
 * The (hard) path to the data file.
 *
 * \return
 * An Eigen matrix with rows/cols determined by data file.
 */
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > CSV2Eigen( std::string file_path ) {

    std::ifstream file_stream( file_path.c_str() );

    if ( !file_stream.good() ) {
        std::string err_str = __func__;
        err_str += "\nCould not open CSV file at location :";
        err_str += file_path;
        throw std::ios_base::failure( err_str );
    } else {
        file_stream.close();
    }

    arma::Mat<T> X;
    X.load( file_path, arma::csv_ascii );
    std::cout << X.n_rows << "x" << X.n_cols << std::endl;

    return Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > >( X.memptr(), X.n_rows, X.n_cols );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > load_from_file (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<T> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }

    return Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > >(values.data(), rows, values.size()/rows);
}

template< typename T >
/*!
 * \brief Compute the standard deviation of a matrix
 *
 * \param mat
 *
 * Matrix to be examined.
 *
 * \return Standard deviation of the matrix
 */
T StdDev( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat ) {

    Eigen::RowVectorXf mean = mat.colwise().mean();
    return ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();
}


template< typename T >
/*!
 * \brief Set the mean of a matrix to 0 and the standard deviation to 1.
 *
 * Note this function is done in place, that is the input matrix is modified.
 *
 * \param mat
 *
 * An n x m matrix to be normalized.
 */
void Normalize_IP ( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Normalize(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template< typename T >
/*!
 * \brief Set the mean of a vector to 0 and the standard deviation to 1.
 *
 * Note this function is done in place.
 *
 * \param mat
 *
 * An n x 1 vector to be normalized.
 */
void Normalize_IP( Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
Eigen::Matrix<T, Eigen::Dynamic, 1 > Normalize(
    const Eigen::Matrix<T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    return (mat.rowwise() - mean).array().rowwise() / std.array();
}

template < typename T >
T eucl_distance( uint n, uint m ) {
    //Indices need to be incremented by one to agree with R's indexing
    return sqrt( (m+1)*(m+1) + (n+1)*(n+1) );
}

template< typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> build_matrix( uint num_rows, uint num_cols, T (*mat_func)(uint,uint) ) {

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat( num_rows, num_cols );

    for( uint i = 0; i < num_rows ; i ++ ) {

        for( uint j = 0; j < num_cols ; j++ ) {
            mat( i, j ) = (*mat_func)( i, j );
        }
    }

    return mat;
}

template< typename T >
void sweep_matrix( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, T (*mat_func)(uint,uint) ) {

    for( uint i = 0; i < mat.rows() ; i ++ ) {

        for( uint j = 0; j < mat.cols() ; j++ ) {
            mat( i, j ) = (*mat_func)( i, j );
        }
    }
}

template < typename T >
typename T::value_type L_infinity_norm( const T& matrix ) {
    return matrix.template lpNorm< Eigen::Infinity >();
}

template < typename T >
typename T::value_type L1_norm( const T& matrix ) {
    return matrix.template lpNorm< 1 >();
}

template < typename T >
typename T::value_type compute_sqr_norm( const T& matrix ) {
    return static_cast< typename T:: value_type >( matrix.squaredNorm() );
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

template <typename T>
T sgn(T val) {
    return static_cast<T>( T(0) < val ) - ( val < T(0) );
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

#endif // FOS_GENERICS_H
