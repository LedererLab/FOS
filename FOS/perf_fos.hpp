#ifndef PERF_FOS_H
#define PERF_FOS_H

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
#include "x_fos.hpp"

namespace hdim {

namespace experimental {

template < typename T >
std::vector< T > PerfX_FOS( SolverType s_type ) {

    std::cout << "Timing FOS for data type: " << get_type_name<T>() << std::endl;

    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/test_data.csv";

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > raw_data = CSV2Eigen< T >( data_set_path );
//    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > raw_data = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >::Random( 2000, 2001 );

    std::cout << "Imported an m = " << raw_data.rows() << " by n = " << raw_data.cols() << " Matrix." << std::endl;

    std::vector < T > timing_results;

    for( unsigned int k = 200; k <= 2000 ; k += 200 ) {

        std::cout << "Testing FOS for a " \
                  << k \
                  << "x" \
                  << k \
                  << "Matrix: \n" \
                  << std::endl;

        Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X = raw_data.block( 0, 1, k, k );
        Eigen::Matrix< T, Eigen::Dynamic, 1 > Y = raw_data.block( 0, 0, k, 1 );

        X_FOS< T > algo_fos;

        auto start = std::chrono::high_resolution_clock::now();

        algo_fos( X, Y, s_type );

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        auto time_taken = ms.count();

        std::cout << "FOS took " << time_taken << " ms." << std::endl;

        timing_results.push_back( time_taken );
    }

    return timing_results;
}

}

}

#endif // PERF_FOS_H
