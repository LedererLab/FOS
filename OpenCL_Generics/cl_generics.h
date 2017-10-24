#ifndef CL_GENERICS_H
#define CL_GENERICS_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
//
// Boost Headers
//
// ViennCL Headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
// CL BLAS
//
// Project Specific Headers
//

template < typename T >
T norm_sqr ( const viennacl::matrix<T>& mat ) {
    T l_2_norm = viennacl::linalg::norm_2( mat );
    return l_2_norm * l_2_norm;
}

template < typename T >
T norm_sqr ( const viennacl::vector<T>& vec ) {
    T l_2_norm = viennacl::linalg::norm_2( vec );
    return l_2_norm * l_2_norm;
}

#endif // CL_GENERICS_H
