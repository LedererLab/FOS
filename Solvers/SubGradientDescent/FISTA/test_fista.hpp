#ifndef TEST_FISTA_HPP
#define TEST_FISTA_HPP

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
#include "fista.hpp"
#include "viennacl_fista.hpp"

template < typename T, typename Solver >
Eigen::Matrix< T, Eigen::Dynamic, 1 > TestFISTA( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > design_matrix,
        Eigen::Matrix< T, Eigen::Dynamic, 1 > predictors,
        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_zero,
        unsigned int number_of_iterations ) {

    Solver fista_test( beta_zero, 0.1 );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > fista_retval = fista_test( design_matrix,
            predictors,
            beta_zero,
            1.0,
            number_of_iterations );

    return fista_retval;
}

template< typename T >
bool approximatelyEqual(T a, T b ) {
    return std::abs(a - b) <= ( (std::abs(a) < std::abs(b) ? std::abs(b) : std::abs(a)) * std::numeric_limits<T>::epsilon());
}

template< typename T >
void PrintDifferences( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& a, const Eigen::Matrix< T, Eigen::Dynamic, 1 >&b ) {

    std::cout.precision(17);

    for( unsigned int i = 0; i < a.rows() ; i++ ) {
        if( !approximatelyEqual( a[i], b[i] ) ) {
            std::cout << "Difference at entry " << i << " detected: " << a[i] << " vs. " << b[i] << std::endl;
        }
    }
}

template < typename T >
void RunFISTATests() {

    for ( unsigned int k = 1000; k <= 5000; k+= 1000 ) {

        unsigned int N = k, P = k;

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >::Random( N , P );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Random( N, 1 );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > W_0 = Eigen::Matrix< T, Eigen::Dynamic, 1 >::Zero( N, 1 );

        std::cout << "Testing FISTA for a "
                  << k
                  << "x"
                  << k
                  << "Matrix: \n"
                  << std::endl;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > cpu_results = TestFISTA< T, hdim::FISTA<T> >( X, Y, W_0, 10 );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > gpu_results = TestFISTA< T, hdim::CL_FISTA<T> >( X, Y, W_0, 10 );

        if( cpu_results ==  gpu_results ) {
            std::cout << "CPU and GPU results agree!" << std::endl;
        } else {
            std::cout << "CPU and GPU results differ" << std::endl;
            PrintDifferences( cpu_results, gpu_results );
        }

    }
}

#endif // TEST_FISTA_HPP
