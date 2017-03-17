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
// Armadillo Headers
//
// Project Specific Headers
#include "OpenCL_Generics/perf_cl_product.h"
#include "ISTA/perf_ista.h"
#include "SPAMS/perf_fista.h"
#include "ISTA/test_ista.h"
#include "SPAMS/test_fista.h"
#include "FOS/test_fos.h"
#include "FOS/test_fos_experimental.h"


int main(int argc, char *argv[]) {

//    RunIstaTests();
//    RunFistaTests();

//    hdim::experimental::TestFOS< float >();
    hdim::TestFOS< float >();

//    std::vector< double > cpu_times;
//    std::vector< double > gpu_times;

//    auto mat_mul_test = MatProdTest( 0, 0 );

//    for( uint k = 2; k <= std::pow( 2, 12 ); k *= 2 ) {

//        std::pair< double, double > data_point = mat_mul_test.Run( k, k );
//        std::pair< double, double > data_point_biased = mat_mul_test.RunBiased( k, k );

//        double gpu_cpu_ratio = data_point.first / data_point.second;
//        double gpu_cpu_ratio_biased = data_point_biased.first / data_point_biased.second;

//        cpu_times.push_back( gpu_cpu_ratio );
//        gpu_times.push_back( gpu_cpu_ratio_biased );

//        cpu_times.push_back( data_point.first );
//        gpu_times.push_back( data_point.second );

//        std::cout << "Ratio of CPU/GPU time: "
//                  << data_point.first / data_point.second
//                  << std::endl;

//    }


}
