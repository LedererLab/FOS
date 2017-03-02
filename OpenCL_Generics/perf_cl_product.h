#ifndef PERF_CL_PRODUCT_H
#define PERF_CL_PRODUCT_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
// Boost Headers
//
// SPAMS Headers
//
// CL BLAS
//
// Project Specific Headers
#include "../Generic/debug.h"
#include "../OpenCL_Base/openclbase.h"

class MatProdTest : public ocl::OpenCLBase {

  public:
    MatProdTest( uint platform_number = 0, uint device_number = 0 );
    ~MatProdTest();

    std::pair< double, double > Run(uint num_rows, uint num_cols);
    std::pair< double, double > RunBiased(uint num_rows, uint num_cols);

  private:

    double CPUMatMul( uint num_rows, uint num_cols );
    double CLMatMulBiased( uint num_rows, uint num_cols );
    double CLMatMul( uint num_rows, uint num_cols );

    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > mat_X;
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > mat_Y;
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > mat_Z;

    cl_int err;
    cl_event event = NULL;

    size_t ld_X;
    cl::Buffer mat_in_X;

    size_t ld_Y;
    cl::Buffer mat_in_Y;

    size_t ld_Z;
    cl::Buffer mat_out_Z;

};

#endif // PERF_CL_PRODUCT_H
