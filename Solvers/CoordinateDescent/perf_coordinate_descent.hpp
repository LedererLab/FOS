#ifndef PERF_COORDINATE_DESCENT_HPP
#define PERF_COORDINATE_DESCENT_HPP

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
#include "../../Generic/debug.hpp"
#include "../../Generic/generics.hpp"
#include "coordinate_descent.hpp"

template < typename T, typename Solver >
int PerfCD( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > design_matrix,
              Eigen::Matrix< T, Eigen::Dynamic, 1 > predictors,
              Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_zero,
              unsigned int number_of_iterations ) {

    Solver cd_test( design_matrix, predictors, beta_zero );

    auto start = std::chrono::high_resolution_clock::now();

    cd_test( design_matrix,
             predictors,
             beta_zero,
             1.0,
             number_of_iterations );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    int time_taken = ms.count();
    \

    return time_taken;
}

template < typename T >
void RunCDPerfs() {

    std::vector<int> standard_results;
    std::vector<int> screened_results;

    for ( unsigned int k = 200; k <= 1000; k+= 200 ) {

        unsigned int N = k, P = k;

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >::Random( N , P );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Random( N, 1 );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > W_0 = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Zero( N, 1 );

        std::cout << "Testing Coordinate Descent for a "
                  << k
                  << "x"
                  << k
                  << "Matrix: \n"
                  << std::endl;

        std::cout << "Testing Standard CD." << std::endl;
        T standard_result = PerfCD< T, hdim::LazyCoordinateDescent<T, hdim::internal::Solver<T> > >( X, Y, W_0, 10 );
        standard_results.push_back( standard_result );

        std::cout << "Testing Screened CD." << std::endl;
        T screened_result = PerfCD< T, hdim::LazyCoordinateDescent<T,hdim::internal::ScreeningSolver<T>> >( X, Y, W_0, 10 );
        screened_results.push_back( screened_result );
    }

    for( unsigned int i = 0; i < standard_results.size() ; i++ ) {
        std::cout << "Standard Timing Results (ms): " << standard_results[i] << " , Screened Timing Results (ms): " << screened_results[i] << std::endl;
    }
}

#endif // PERF_COORDINATE_DESCENT_HPP
