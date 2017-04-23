#ifndef TEST_FOS_EXPERIMENTAL_H
#define TEST_FOS_EXPERIMENTAL_H

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
#include "fos_imperative.h"
#include "../Generic/generics.h"
#include "x_fos.h"

namespace hdim {

namespace experimental {

template < typename T >
std::vector< T > TestFOS() {

    std::cout << "Running experimental::FOS test for data type: " << get_type_name<T>() << std::endl;
    std::cout << "Using experimental version of FOS with ISTA." << std::endl;

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

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = raw_data.block( 0, 1, k, k );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = raw_data.block( 0, 0, k, 1 );

        std::cout << "Froebenius squared norm of X " \
                  << X.squaredNorm()\
                  << " Froebenius squared norm of Y "\
                  << Y.squaredNorm() \
                  << std::endl;

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Beta;

        TIME_IT( Beta = experimental::Imp_FOS< T >( X, Y ); );

        T sqr_norm = Beta.squaredNorm();
        std::cout << "Froebenius norm squared of Beta tilde, r tilde: " << sqr_norm << std::endl;
        std::cout << std::endl;

        sqr_norm_results.push_back( sqr_norm );
    }

    return sqr_norm_results;
}

template < typename T >
std::vector< T > TestX_FOS() {

    std::cout << "Running X_FOS test for data type: " << get_type_name<T>() << std::endl;

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

        X_FOS< T > algo_fos;
        TIME_IT( algo_fos( X, Y ); );

        std::cout << "Stopping index: " << algo_fos.ReturnOptimIndex() << std::endl;
        T sqr_norm = algo_fos.ReturnCoefficients().squaredNorm();
        std::cout << "Froebenius norm squared of Beta tilde, r tilde: " << sqr_norm << std::endl;
        std::cout << std::endl;

        sqr_norm_results.push_back( sqr_norm );
    }

    return sqr_norm_results;
}

}

}

#endif // TEST_FOS_EXPERIMENTAL_H
