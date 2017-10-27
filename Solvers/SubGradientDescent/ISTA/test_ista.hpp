#ifndef TEST_ISTA_H
#define TEST_ISTA_H

// C System-Headers
//
// C++ System headers
#include <chrono>
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
#include "viennacl_ista.hpp"

template < typename T, typename Solver >
T TestISTA( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > design_matrix,
            Eigen::Matrix< T, Eigen::Dynamic, 1 > predictors,
            Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_zero,
            unsigned int number_of_iterations ) {

    Solver ista_test( 0.1 );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > ista_retval = ista_test( design_matrix,
            predictors,
            beta_zero,
            1.0,
            number_of_iterations );

    return ista_retval.squaredNorm();
}

template < typename T >
void RunIstaTests() {

    std::vector<T> cpu_results;
    std::vector<T> gpu_results;

    for ( unsigned int k = 200; k <= 1000; k+= 200 ) {

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
        T cpu_result = TestISTA< T, hdim::ISTA<T> >( X, Y, W_0, 10 );
        cpu_results.push_back( cpu_result );

        std::cout << "Testing GPU ISTA" << std::endl;
        T gpu_result = TestISTA< T, hdim::vcl::ISTA<T> >( X, Y, W_0, 10 );
        gpu_results.push_back( gpu_result );
    }

    for( unsigned int i = 0; i < cpu_results.size() ; i++ ) {
        std::cout << "CPU Results: " << cpu_results[i] << " , GPU Results: " << gpu_results[i] << std::endl;
    }
}

#endif // TEST_ISTA_H
