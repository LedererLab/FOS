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


int main(int argc, char *argv[]) {

//    auto x_fos = hdim::experimental::PerfX_FOS< double >();
//    auto old_fos = hdim::PerfFOS< double >();
//    auto old_fos = hdim::experimental::PerfFOS< double >();

    auto x_fos = hdim::experimental::TestX_FOS< double >();
//    auto old_fos = hdim::experimental::TestFOS< double >();

//    auto fos_results_w_ista = hdim::experimental::TestFOS< double >();
//    auto x_fos = hdim::experimental::TestX_FOS< double >();
//    auto old_fos = hdim::TestFOS< double >();

//    std::vector< double > ratio_results;

//    for( uint i = 0 ; i < x_fos.size() ; i++ ) {

//        auto ratio = x_fos.at(i) / old_fos.at(i);
//        ratio_results.push_back( ratio );
//    }


//    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
//                                                  "Timing Results Ratio",
//                                                  "Row Size / 200",
//                                                  "Execution time ( X_FISTA / FISTA ).",
//                                                  "Ratio of FOS timing with X_FISTA v FISTA",
//                                                  "/home/bephillips2/");

//    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
//                                                  "L2 Norm of Beta Ratio",
//                                                  "Row Size / 200",
//                                                  "L2 Norm of Beta ( X_FISTA / FISTA ).",
//                                                  "Ratio of FOS results X_FISTA v. X_ISTA",
//                                                  "/home/bephillips2/");

}
