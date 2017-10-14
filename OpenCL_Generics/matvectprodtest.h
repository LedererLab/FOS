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
#include "../OpenCL_Base/openclbase.h"

class MatVectProdTest : public ocl::OpenCLBase {

  public:
    MatVectProdTest(uint platform_number = 0, uint device_number = 0);
    ~MatVectProdTest();

    std::pair< double, double > Run(uint num_rows, uint num_cols, uint num_iterations);
    std::pair< double, double > RunBiased(uint num_rows, uint num_cols, uint num_iterations);

  private:

    double CPUMatMul( uint num_rows, uint num_cols, uint num_iterations );
    double CLMatMulBiased( uint num_rows, uint num_cols, uint num_iterations);
    double CLMatMul( uint num_rows, uint num_cols, uint num_iterations );

    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > mat_A;
    Eigen::Matrix< float, Eigen::Dynamic, 1 > vec_X;
    Eigen::Matrix< float, Eigen::Dynamic, 1 > vec_Y;

    cl_int err;
    cl_event event = NULL;

    size_t off;

    size_t offA;
    cl::Buffer mat_in_A;

    size_t offX;
    cl::Buffer vec_in_X;

    size_t offY;
    cl::Buffer vec_out_Y;

    int incx = 1;
    int incy = 1;
    size_t lda;

};


#endif // MATVECTPRODTEST_H
