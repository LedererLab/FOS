#ifndef TEST_FISTA_H
#define TEST_FISTA_H

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
#include "../Generic/algorithm.h"

void TestFistaFlat( uint num_rows, uint num_cols ) {

    auto X = build_matrix<float>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);
    auto W_0 = Eigen::Matrix< float, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    float lambda = 1.0;

    auto spams_retval =  FistaFlat< float >( Y, X, W_0, 0.5*lambda );

    std::cout << "fistaFlat result:\n" << spams_retval.squaredNorm() << std::endl;
}

void RunFistaTests() {

    for ( uint k = 200; k <= 2000; k+= 200 ) {

        std::cout << "Testing fistaFlat for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
//                  << build_matrix<float>( k, k, &eucl_distance )
                  << std::endl;

        TestFistaFlat( k, k );
    }
}

#endif // TEST_FISTA_H
