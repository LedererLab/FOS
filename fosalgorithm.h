#ifndef FOSALGORITHM_H
#define FOSALGORITHM_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// SPAMS Headers
#include "linalg.h" // AbstractMatrix and Matrix
#include "spams.h"
// Armadillo Headers
#include <armadillo>
// Project Specific Headers
//

/*! \file Functions that provide an interface between Eigen and Spams linear algebra libraries.
 *
 */

template < typename T, uint m, uint n >
Eigen::Matrix< T, m, n > Spams2EigenMat ( const Matrix<T>* spams_mat ) {
    auto M = Eigen::Map< Eigen::Matrix< T, n, m, Eigen::ColMajor> >( spams_mat->rawX() );
    return M;
}

template < typename T, uint m, uint n >
Eigen::Matrix< T, m, n > Spams2EigenMat ( Matrix<T>* spams_mat ) {

    auto M = Eigen::Map< Eigen::Matrix< T, n, m, Eigen::ColMajor> >( spams_mat->rawX() );
    return M;
}


template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Spams2EigenMat ( Matrix<T>* spams_mat ) {

    uint num_cols = spams_mat->n();
    uint num_rows = spams_mat->m();

    auto M = Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >( spams_mat->rawX(), num_cols, num_rows );
    return M;
}

template < typename T, uint m, uint n >
Matrix<T>* Eigen2SpamsMat ( const Eigen::Matrix< T, n, m >& eigen_mat ) {

    auto eigen_mat_size = eigen_mat.cols() * eigen_mat.rows();

    T* non_const_mat_data = new T[eigen_mat_size]; // create a new buffer
    auto mat_data = eigen_mat.data();

    std::copy( mat_data, mat_data + eigen_mat_size, non_const_mat_data); // copy the data

    auto spams_mat = new Matrix<T> ( non_const_mat_data, m, n );

    return spams_mat;
}

template < typename T >
Matrix<T>* Eigen2SpamsMat ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eigen_mat ) {

    auto m = eigen_mat.rows();
    auto n = eigen_mat.cols();

    auto eigen_mat_size = m * n ;

    T* non_const_mat_data = new T[eigen_mat_size]; // create a new buffer
    auto mat_data = eigen_mat.data();

    std::copy( mat_data, mat_data + eigen_mat_size, non_const_mat_data); // copy the data

    auto spams_mat = new Matrix<T> ( non_const_mat_data, m, n );

    return spams_mat;
}

template < typename T, uint m, uint n >
AbstractMatrixB<T> Eigen2SpamsAbstractMatB ( const Eigen::Matrix< T, n, m >& eigen_mat ) {
    return AbstractMatrixB<T>( eigen_mat.data(), m, n );
}

//Will need to be deleted by user
char* str_to_c_ptr( std::string& str ) {

    char * writable = new char[str.size() + 1];
    std::copy(str.begin(), str.end(), writable);
    writable[str.size()] = '\0'; // don't forget the terminating 0

    return writable;
}

namespace internal {

template < typename T >
Matrix<T>* FistaFlat( Matrix<T>* Y, Matrix<T>* X, Matrix<T>* Omega_0, const T lambda_1 ) {

    uint num_cols = Omega_0->n();
    uint num_rows = Omega_0->m();

    //Initialize alpha
    auto W = new Matrix<T> ( num_rows, num_cols );
    W->setZeros();

    //Initialize groups
    auto groups = new Vector<int>( num_rows );
    groups->setZeros();

    //Initialize num_threads
    int num_threads = omp_get_max_threads();

    auto inner_weights = (Vector< double > *) 0;

    char regul[] = "l1";
    char loss[] = "square";
    char log_name[] = "";

    //Return value is optimization info which we do not need
    //We are interested in 'W' which is implicitly modified
    _fistaFlat(Y, //X
               X, //D
               Omega_0, //alpha0
               W, // alpha
               groups, // groups
               num_threads, // num_threads
               1, // mat_it
               static_cast<T>( 0.1 ), //L0
               false, //fixed_step
               static_cast<T>( 1.5 ), // gamma
               lambda_1, //lambda_
               static_cast<T>( 1.0 ), //delta
               static_cast<T>( 0.0 ), //lambda2
               static_cast<T>( 0.0 ), //lambda3
               static_cast<T>( 1.0 ), //a
               static_cast<T>( 0.0 ), //b
               static_cast<T>( 1.0 ), //c
               static_cast<T>( 0.000001 ), //tol
               100, //it0
               1000, //max_iter_backtracking
               false, //compute_gram
               false, //lin_admm
               false, //admm
               false, //intercept
               false, //resetflow
               regul, //name_regul
               loss, //name_loss
               false, //verbose
               false, //pos
               false, //clever
               false, //log
               true, //ista
               false, //subgrad
               log_name, //logName
               false, //is_inner_weights
               inner_weights, //inner_weights
               1, //size_group
               true, //sqrt_step
               false, //transpose
               0 //linesearch_mode
              );
    delete Y;
    delete X;
    delete Omega_0;

//    delete W;
    delete groups;
    delete inner_weights;

    return W;
}

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic> FistaFlat( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic> Y,\
        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic> X, \
        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic> Omega_0, \
        const T lambda_1 ) {


    auto spams_Y = Eigen2SpamsMat<T>( Y );
    auto spams_X = Eigen2SpamsMat<T>( X );
    auto spams_omega =  Eigen2SpamsMat<T>( Omega_0 );

    auto spams_ret_val = internal::FistaFlat< T >( spams_Y, spams_X, spams_omega, lambda_1 );

    auto ret_val = Spams2EigenMat<T>( spams_ret_val );

    return ret_val.transpose();
}

#endif // FOSALGORITHM_H
