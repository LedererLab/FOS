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
#include "fos.h"

namespace hdim {

template < typename T >
std::vector< T > TestFOS() {

    std::cout << "Running FOS test for data type: " << get_type_name<T>() << std::endl;

    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/test_data.csv";

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > raw_data = CSV2Eigen< T >( data_set_path );

    std::cout << "Imported an m = " << raw_data.rows() << " by n = " << raw_data.cols() << " Matrix." << std::endl;

    std::vector < T > sqr_norm_results;

    for( uint k = 200; k <= 2000 ; k += 200 ) {

        std::cout << "Testing FOS for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
                  << std::endl;

        auto X = raw_data.block( 0, 1, k, k );

        auto Y = raw_data.block( 0, 0, k, 1 );

        std::cout << "Froebenius squared norm of X " \
                  << X.squaredNorm()\
                  << " Froebenius squared norm of Y "\
                  << Y.squaredNorm() \
                  << std::endl;

        FOS< T > algo_fos ( X, Y );
        TIME_IT( algo_fos.Algorithm(); );

        std::cout << "Stopping index: " << algo_fos.ReturnOptimIndex() << std::endl;
        T sqr_norm = algo_fos.ReturnCoefficients().squaredNorm();
        std::cout << "Froebenius norm squared of Beta tilde, r tilde: " << sqr_norm << std::endl;
        std::cout << std::endl;

        sqr_norm_results.push_back( sqr_norm );
    }

    return sqr_norm_results;
}

}

#endif // TEST_FOS_H
