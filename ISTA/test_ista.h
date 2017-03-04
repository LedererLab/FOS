#ifndef TEST_ISTA_H
#define TEST_ISTA_H


// C System-Headers
//
// C++ System headers
#include <cmath>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// SPAMS Headers
//
// Armadillo Headers
//
// Project Specific Headers
#include "../Generic/debug.h"
#include "../Generic/generics.h"
#include "ista.h"

void TestIsta( uint num_rows, uint num_cols ) {

    auto X = build_matrix<float>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);
    auto W_0 = Eigen::Matrix< float, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    float lambda = 1.0;

    auto ista_retval = ISTA< float >( X, Y, W_0, 1, 0.1, 0.5*lambda );

    std::cout << "ISTA result:\n" << ista_retval << std::endl;
}

void RunIstaTests() {

    for ( uint k = 2; k <= 10; k++ ) {

        std::cout << "Testing ISTA for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
                  << build_matrix<float>( k, k, &eucl_distance ) \
                  << std::endl;

        TestIsta( k, k );
    }
}

#endif // TEST_ISTA_H
