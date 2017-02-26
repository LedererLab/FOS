#ifndef PERF_FISTA_H
#define PERF_FISTA_H

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

void PerfFista( uint num_rows, uint num_cols ) {

    auto X = build_matrix<double>( num_rows, num_cols, &eucl_distance );
    auto Y = X.col(0);
    auto W_0 = Eigen::Matrix< double, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    double lambda = 1.0;

    TIME_IT( FistaFlat< double >( Y, X, W_0, 0.5*lambda ); );

}

void RunFistaPerfTests() {

    for ( uint k = 200; k <= 2000; k += 200 ) {

        std::cout << "Testing FISTA for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix:" \
                  << std::endl;

        PerfFista( k, k );
    }
}

#endif // PERF_FISTA_H
