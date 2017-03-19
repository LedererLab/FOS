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
#include "OpenCL_Generics/perf_cl_product.h"
#include "ISTA/perf_ista.h"
#include "SPAMS/perf_fista.h"
#include "ISTA/test_ista.h"
#include "SPAMS/test_fista.h"
#include "FOS/test_fos.h"
#include "FOS/test_fos_experimental.h"
#include "FOS/perf_fos.h"
#include "FOS/perf_fos_experimental.h"


int main(int argc, char *argv[]) {

    auto fos_results_w_ista = hdim::experimental::PerfFOS< double >();
    auto fos_results_w_fista = hdim::PerfFOS< double >();

//    auto fos_results_w_ista = hdim::experimental::TestFOS< double >();
//    auto fos_results_w_fista = hdim::TestFOS< double >();

    std::vector< double > ratio_results;

    for( uint i = 0 ; i < fos_results_w_ista.size() ; i++ ) {

        auto ratio = fos_results_w_fista.at(i) / fos_results_w_ista.at(i);
        ratio_results.push_back( ratio );
    }

    jaspl::plot< std::vector< double > >( ratio_results, "Ratio of FOS timing w/ ISTA v. FISTA" );

//    jaspl::plot< std::vector< double > >( fos_results_w_ista,
//                 fos_results_w_fista,
//                 "FOS w/ ISTA",
//                 "FOS w/ FISTA",
//                 "Comparison of FOS implementations");

}
