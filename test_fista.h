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
#include "fos_debug.h"
#include "fos_generics.h"
#include "fosalgorithm.h"

double eucl_distance( uint n, uint m ) {
    //Indices need to be incremented by one to agree with R's indexing
    return std::sqrt( (m+1)*(m+1) + (n+1)*(n+1) );
}

void TestFistaFlat( uint num_rows, uint num_cols ) {

    auto X = build_matrix<double>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);
    auto W_0 = Eigen::Matrix< double, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    double lambda = 1.0;

    auto spams_retval =  FistaFlat< double >( Y, X, W_0, 0.5*lambda );

    std::cout << "fistaFlat result:\n" << spams_retval << std::endl;
}

void RunFistaTests() {

    for ( uint k = 2; k <= 10; k++ ) {

        std::cout << "Testing fistaFlat for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
                  << build_matrix<double>( k, k, &eucl_distance ) \
                  << std::endl;

        TestFistaFlat( k, k );
    }
}

#endif // TEST_FISTA_H
