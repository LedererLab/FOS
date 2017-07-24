#ifndef TEST_X_FOS_HPP
#define TEST_X_FOS_HPP

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
// FISTA Headers
//
// Project Specific Headers
#include "x_fos.hpp"

namespace hdim {

template < typename T >
void TestXFOS( unsigned int N, unsigned int P, SolverType s_type ) {

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >::Random( N , P );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Random( N, 1 );

    X_FOS<T> fos;
    TIME_IT( fos( X, Y, s_type ); );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > fos_fit = fos.ReturnCoefficients();
}

}

#endif // TEST_X_FOS_HPP
