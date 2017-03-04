#ifndef PERF_ISTA_H
#define PERF_ISTA_H

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

void PerfIsta( uint num_rows, uint num_cols ) {

    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > X = build_matrix<double>( num_rows, num_cols, &eucl_distance );
    Eigen::Matrix< double, Eigen::Dynamic, 1 > Y = X.col(0);
    Eigen::Matrix< double, Eigen::Dynamic, 1 > W_0 = Eigen::Matrix< double, Eigen::Dynamic, 1 > ( num_rows, 1 );
    W_0.setZero();

    double lambda = 1.0;

    TIME_IT( ISTA< double >( X, Y, W_0, 1, 0.1, 0.5*lambda ); );

}

void RunIstaPerfTests() {

    for ( uint k = 200; k <= 2000; k += 200 ) {

        std::cout << "Testing ISTA for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix:" \
                  << std::endl;

        PerfIsta( k, k );
    }
}

#endif // PERF_ISTA_H
