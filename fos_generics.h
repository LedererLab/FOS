#ifndef FOS_GENERICS_H
#define FOS_GENERICS_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
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
T CSV2Eigen( std::string file_path ) {

    std::ifstream file_stream( file_path.c_str() );

    if ( !file_stream.good() ) {
        std::string err_str = __func__;
        err_str += "\nCould not open CSV file at location :";
        err_str += file_path;
        throw std::ios_base::failure( err_str );
    } else {
        file_stream.close();
    }

    arma::mat X;
    X.load( file_path, arma::csv_ascii );
    std::cout << X.n_rows << "x" << X.n_cols << std::endl;

    return Eigen::Map<const T>( X.memptr(), X.n_rows, X.n_cols );

}

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
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
void Normalize( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
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
void Normalize( Eigen::Matrix< T, Eigen::Dynamic, 1 >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}


#endif // FOS_GENERICS_H