#ifndef MATVECTPRODTEST_H
#define MATVECTPRODTEST_H


// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// Boost Headers
//
// SPAMS Headers
//
// CL BLAS
//
// Project Specific Headers
#include "../Generic/debug.hpp"
#include "../OpenCL_Base/openclbase.hpp"

class MatVectProdTest : public ocl::OpenCLBase {

  public:
    MatVectProdTest( unsigned int platform_number = 0, unsigned int device_number = 0 );
    ~MatVectProdTest();

    std::pair< double, double > Run(unsigned int num_rows, unsigned int num_cols);
    std::pair< double, double > RunBiased(unsigned int num_rows, unsigned int num_cols);

  private:

    double CPUMatMul( unsigned int num_rows, unsigned int num_cols );
    double CLMatMulBiased( unsigned int num_rows, unsigned int num_cols );
    double CLMatMul( unsigned int num_rows, unsigned int num_cols );

    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > mat_X;
    Eigen::Matrix< float, Eigen::Dynamic, 1 > mat_Y;
    Eigen::Matrix< float, Eigen::Dynamic, 1 > mat_Z;

    cl_int err;
    cl_event event = NULL;

    size_t ld_X;
    cl::Buffer mat_in_X;

    size_t ld_Y;
    cl::Buffer mat_in_Y;

    size_t ld_Z;
    cl::Buffer mat_out_Z;

};


#endif // MATVECTPRODTEST_H
