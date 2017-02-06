#ifndef TEST_FOS_H
#define TEST_FOS_H


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
#include "fos_debug.h"
#include "fos_generics.h"
#include "fos.h"
#include "fosalgorithm.h"

void TestFOS( uint num_rows, uint num_cols ) {

    auto X = build_matrix<double>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);

    FOS< double > algo_fos ( X, Y );
    algo_fos.Algorithm();
}

void RunFOSTests() {

    for ( uint k = 2; k <= 20; k++ ) {

        std::cout << "Testing FOS for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
                  << std::endl;

        TestFOS( k, k );
    }
}

#endif // TEST_FOS_H
