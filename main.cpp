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
//
// Project Specific Headers
#include "FOS/test_x_fos.hpp"
#include "Solvers/SubGradientDescent/ISTA/test_ista.hpp"
#include "Solvers/SubGradientDescent/ISTA/perf_ista.hpp"

int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

    RunIstaPerfs<float>();
//    RunIstaTests< float >();
//    hdim::TestXFOS<double>( 200, 500, hdim::SolverType::cd );

}
