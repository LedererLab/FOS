#ifndef PERF_ISTA_H
#define PERF_ISTA_H

// C System-Headers
//
// C++ System headers
#include <cmath>
// Eigen Headers
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// SPAMS Headers
//
// Armadillo Headers
//
// Project Specific Headers
#include "../../../Generic/debug.hpp"
#include "../../../Generic/generics.hpp"
#include "ista.hpp"
#include "viennacl_ista.h"

template < typename T, typename Solver >
int PerfISTA( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > design_matrix,
            Eigen::Matrix< T, Eigen::Dynamic, 1 > predictors,
            Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_zero,
            unsigned int number_of_iterations ) {

    Solver ista_test( 0.1 );

    auto start = std::chrono::high_resolution_clock::now();

    ista_test( design_matrix,
            predictors,
            beta_zero,
            1.0,
            number_of_iterations );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    int time_taken = ms.count();\

    return time_taken;
}

template < typename T >
void RunIstaPerfs() {

    std::vector<int> cpu_results;
    std::vector<int> gpu_results;

    for ( unsigned int k = 1000; k <= 5000; k+= 1000 ) {

        unsigned int N = k, P = k;

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >::Random( N , P );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Random( N, 1 );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > W_0 = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Zero( N, 1 );

        std::cout << "Testing ISTA for a "
                  << k
                  << "x"
                  << k
                  << "Matrix: \n"
                  << std::endl;

        std::cout << "Testing CPU ISTA" << std::endl;
        T cpu_result = PerfISTA< T, hdim::ISTA<T> >( X, Y, W_0, 10 );
        cpu_results.push_back( cpu_result );

        std::cout << "Testing GPU ISTA" << std::endl;
        T gpu_result = PerfISTA< T, hdim::vcl::ISTA<T> >( X, Y, W_0, 10 );
        gpu_results.push_back( gpu_result );
    }

    for( unsigned int i = 0; i < cpu_results.size() ; i++ ) {
        std::cout << "CPU Timing Results (ms): " << cpu_results[i] << " , GPU Timing Results (ms): " << gpu_results[i] << std::endl;
    }
}

#endif // PERF_ISTA_H
