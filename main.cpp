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
#include "Solvers/CoordinateDescent/perf_coordinate_descent.hpp"
#include "Solvers/SubGradientDescent/ISTA/test_ista.hpp"
#include "Solvers/SubGradientDescent/ISTA/perf_ista.hpp"
#include "Solvers/SubGradientDescent/FISTA/test_fista.hpp"
#include "Solvers/SubGradientDescent/FISTA/perf_fista.hpp"

int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

//    RunCDPerfs<float>();

    RunIstaPerfs<float>();
//    RunIstaPerfs<double>();

//    RunISTATests< float >();
//    RunISTATests< double >();

//    RunFISTATests< float >();
//    RunFISTATests< double >();

    RunFISTAPerfs<float>();
//    RunFISTAPerfs<double>();

//    hdim::TestXFOS< float >( 200, 500, hdim::SolverType::cd );
//    hdim::TestXFOS< float >( 200, 500, hdim::SolverType::fista );
    hdim::TestXFOS< float >( 3000, 3000, hdim::SolverType::cl_fista );

}
