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

    (void)argc;
    (void)argv;

//    auto x_fos = hdim::experimental::PerfX_FOS< double >();
//    auto old_fos = hdim::PerfFOS< double >();

    auto x_fos = hdim::experimental::TestX_FOS< double >();
    auto old_fos = hdim::TestFOS< double >();

    std::vector< double > ratio_results;

    for( uint i = 0 ; i < x_fos.size() ; i++ ) {
        auto ratio = x_fos.at(i) / old_fos.at(i);
        ratio_results.push_back( ratio );
    }


    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
                                                  "Timing Results Ratio",
                                                  "Row Size / 200",
                                                  "Execution time X FOS w CD v. FOS w ISTA (ms).",
                                                  "Ratio of X FOS w CD v. FOS w ISTA",
                                                  "/home/bephillips2/");

//    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
//                                                  "L2 Norm of Beta Ratio",
//                                                  "Row Size / 200",
//                                                  "L2 Norm of Beta ( X FOS w CD v. FOS w ISTA ).",
//                                                  "Ratio of results X FOS w CD v. FOS w ISTA",
//                                                  "/home/bephillips2/");

}
