#ifndef TEST_ISTA_H
#define TEST_ISTA_H


// C System-Headers
//
// C++ System headers
#include <cmath>
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
// Boost Headers
//
// SPAMS Headers
//
// Armadillo Headers
//
// Project Specific Headers
#include "../Generic/debug.hpp"
#include "../Generic/generics.hpp"
#include "ista.hpp"

void TestIsta( unsigned int num_rows, unsigned int num_cols ) {

    auto X = build_matrix<float>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);
    auto W_0 = Eigen::Matrix< float, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    float lambda = 1.0;

    Eigen::Matrix< float, Eigen::Dynamic, 1 > ista_retval = ISTA< float >( X, Y, W_0, 10, 0.1, lambda );

    std::cout << "ISTA result:\n" << ista_retval.squaredNorm() << std::endl;
}

void RunIstaTests() {

    for ( unsigned int k = 200; k <= 2000; k+= 200 ) {

        std::cout << "Testing ISTA for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
//                  << build_matrix<float>( k, k, &eucl_distance )
                  << std::endl;

        TestIsta( k, k );
    }
}

#endif // TEST_ISTA_H
