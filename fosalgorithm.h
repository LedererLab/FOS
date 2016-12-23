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
#include <spams/linalg/linalg.h> // AbstractMatrix and Matrix
// Project Specific Headers
//

template < typename T, uint m, uint, n >
Eigen::Matrix< T, m, n > Spams2EigenMat ( const Matrix<T>& spams_mat ) {
    //
}

template < typename T, uint m, uint, n >
Matrix Eigen2SpamsMat ( const Eigen::Matrix< T, n, m >& eigen_mat ) {
    //
}

template < typename T, uint m, uint, n >
AbstractMatrixB Eigen2SpamsAbstractMatB ( const Eigen::Matrix< T, n, m >& eigen_mat ) {
    //
}

#endif // FOSALGORITHM_H
