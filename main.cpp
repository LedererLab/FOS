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
#include "../JASPL/jPlot/jplot.h"


int main(int argc, char *argv[]) {

//    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/hdim/App/src/DataSets/riboflavin_t_no_header.csv";

//    Eigen::MatrixXd raw_data = CSV2Eigen< Eigen::MatrixXd >( data_set_path );

//    auto X = raw_data.block( 0, 1, raw_data.rows(), raw_data.cols() - 1 );
//    auto Y = raw_data.col(0);

    std::vector< double > cpu_times;
    std::vector< double > gpu_times;

    auto mat_mul_test = MatProdTest( 0, 0 );

    for( uint k = 2; k <= std::pow( 2, 12 ); k *= 2 ) {

        std::pair< double, double > data_point = mat_mul_test.Run( k, k );
        std::pair< double, double > data_point_biased = mat_mul_test.RunBiased( k, k );

        double gpu_cpu_ratio = data_point.first / data_point.second;
        double gpu_cpu_ratio_biased = data_point_biased.first / data_point_biased.second;

        cpu_times.push_back( gpu_cpu_ratio );
        gpu_times.push_back( gpu_cpu_ratio_biased );

//        cpu_times.push_back( data_point.first );
//        gpu_times.push_back( data_point.second );

//        std::cout << "Ratio of CPU/GPU time: "
//                  << data_point.first / data_point.second
//                  << std::endl;

    }

    jaspl::plot( cpu_times, gpu_times, "CPU/GPU Time", "CPU/GPU Time ( w/o Load/Recall )", "CPU vs. GPU Timing Ratio" );
//    jaspl::plot( cpu_times, gpu_times, "Intel(R) Core(TM) i7-4700HQ CPU 2.40 GHz", "GeForce GTX 860M", "CPU vs. GPU Matrix Product" );

}
