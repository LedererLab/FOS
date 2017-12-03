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


int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

    hdim::TestXFOS< float >( 200, 500, hdim::SolverType::cd );
    hdim::TestXFOS< float >( 200, 500, hdim::SolverType::fista );

}
