// C System-Headers
//
// C++ System headers
//
// Eigen Headers
//
// Boost Headers
//
// SPAMS Headers
//
// JASPL
#include "../JASPL/jPlot/jplot.h"
// Project Specific Headers
#include "FOS/test_fos.h"
#include "FOS/test_fos_experimental.h"
#include "FOS/perf_fos.h"
#include "FOS/perf_fos_experimental.h"

/*! \mainpage C++ Implementation of packages developed by the
 * Lederer and Hauser HDIM Group
 *
 * \section intro_sec Introduction
 *
 * HDIM provides fast methods to perform high-dimensional linear regression -- most notably the FOS ( Fast and Optimal Selection ) method for minimizing the LASSO objective function.
 *
 * \section Base Dependencies
 *   \li Eigen3
 *
 * \section Dependencies for OpenCL Acceleration
 *  \li OpenCL SDK ( AMD-APP, CUDA, Intel Beignet )
 *  \li clBLAS
 *
 * \section Dependencies for Python Wrappers
 *  \li Python 2.7 Installation
 *  \li Simplified Wrapper and Inteface Generator ( SWIG )
 *
 *
 */

int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

    auto x_fos = hdim::experimental::PerfX_FOS< double >();
    auto old_fos = hdim::PerfFOS< double >();

//    auto x_fos = hdim::experimental::TestX_FOS< double >();
//    auto old_fos = hdim::TestFOS< double >();

    std::vector< double > ratio_results;

    for( uint i = 0 ; i < x_fos.size() ; i++ ) {

        auto ratio = x_fos.at(i) / old_fos.at(i);
        ratio_results.push_back( ratio );
    }


    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
                                                  "Timing Results Ratio",
                                                  "Row Size / 200",
                                                  "Execution time ( X FOS v. FOS ).",
                                                  "Ratio of FOS timing, X FOS w ISTA_{OPT} v. FOS",
                                                  "/home/bephillips2/");

//    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
//                                                  "L2 Norm of Beta Ratio",
//                                                  "Row Size / 20",
//                                                  "L2 Norm of Beta ( X FOS  w FISTA v. FOS ).",
//                                                  "Ratio of results X FOS  w FISTA v. FOS",
//                                                  "/home/bephillips2/");

}
