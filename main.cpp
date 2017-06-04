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
#include "FOS/test_fos.hpp"
#include "FOS/test_fos_experimental.hpp"
#include "FOS/perf_fos.hpp"

int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

    auto x_fos_cd = hdim::experimental::TestX_FOS< float >( hdim::SolverType::cd );
//    auto x_fos_fista = hdim::experimental::TestX_FOS< float >( hdim::SolverType::ista );

////    auto x_fos_cd = hdim::experimental::PerfX_FOS< float >( hdim::SolverType::ista );
////    auto x_fos_fista = hdim::experimental::PerfX_FOS< float >( hdim::SolverType::fista );

//    std::vector< double > ratio_results;

//    for( unsigned int i = 0 ; i < x_fos_cd.size() ; i++ ) {
//        auto ratio = x_fos_cd.at(i) / x_fos_fista.at(i);
//        ratio_results.push_back( ratio );
//    }


////    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
////                                                  "Timing Results Ratio",
////                                                  "Row Size / 200",
////                                                  "Execution time X FOS w ISTA v. X FOS w FISTA (ms).",
////                                                  "Ratio of time, X FOS w ISTA v. X FOS w FISTA ",
////                                                  "/home/bephillips2/");

//    jaspl::plot_to_disk< std::vector< double > >( ratio_results,
//                                                  "L2 Norm of Beta Ratio",
//                                                  "Row Size / 200",
//                                                  "L2 Norm of Beta ( X FOS w CD v. X FOS w ISTA ).",
//                                                  "Ratio of results X FOS w CD v. X FOS w ISTA ",
//                                                  "/home/bephillips2/");

}
