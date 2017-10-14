#include "matvectprodtest.h"

// C System-Headers
//
// C++ System headers
#include <chrono>
// Eigen Headers
//
// Boost Headers
//
// SPAMS Headers
//
// CL BLAS
#include <clBLAS.h>
// Project Specific Headers
#include "perf_cl_product.h"
#include "../Generic/debug.hpp"

MatVectProdTest::MatVectProdTest(uint platform_number, uint device_number ) : OpenCLBase( platform_number, device_number ) {

    err = clblasSetup();
}

MatVectProdTest::~MatVectProdTest() {

    clblasTeardown();

}

double MatVectProdTest::CPUMatMul( uint num_rows, uint num_cols, uint num_iterations ) {

    std::cout << "Testing matrix product using Eigen3." << std::endl;

    mat_A = Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic >::Random( num_rows, num_cols );
    vec_X = Eigen::Matrix< float, Eigen::Dynamic, 1 >::Random( num_cols, 1 );
    vec_Y.setZero( num_rows, 1 );

    auto start = std::chrono::high_resolution_clock::now();

    for( uint i = 0; i < num_iterations ; i++ ) {

        vec_Y = mat_A*vec_X;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;
    \

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << vec_Y.squaredNorm() << std::endl;

    return time_taken;

}

double MatVectProdTest::CLMatMul(uint num_rows, uint num_cols , uint num_iterations) {

    std::cout << "Testing matrix product using clBLAS." << std::endl;

    mat_A = Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic >::Random( num_rows, num_cols );
    vec_Y = Eigen::Matrix< float, Eigen::Dynamic, 1 >::Random( num_cols, 1 );
    vec_Y.setZero( num_rows, 1 );


    off  = 1;
    offA = num_rows + off;   /* M + off */
    offX = off;       /* off */
    offY = off;       /* off */

    incx = 1;
    incy = 1;
    lda = num_cols;        /* i.e. lda = N */

    mat_in_A = cl::Buffer( context,
                           CL_MEM_READ_ONLY,
                           num_rows * num_cols * sizeof( float ),
                           NULL, &err );

    //cl::Buffer ( context, CL_MEM_READ_WRITE, processed_bytes, err_ptr );
    vec_in_X = cl::Buffer ( context,
                            CL_MEM_READ_ONLY,
                            num_cols * 1 * sizeof( float ),
                            NULL, &err );

    vec_out_Y = cl::Buffer( context,
                            CL_MEM_READ_WRITE,
                            num_rows * 1 * sizeof( float ),
                            NULL, &err );

    auto start = std::chrono::high_resolution_clock::now();

    command_queue.enqueueWriteBuffer( mat_in_A, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_A.data() );
    command_queue.enqueueWriteBuffer( vec_in_X, CL_TRUE, 0, num_cols * 1 * sizeof( float ), vec_X.data() );
    command_queue.enqueueWriteBuffer( vec_out_Y, CL_TRUE, 0, num_rows * 1 * sizeof( float ), vec_Y.data() );

    for( uint i = 0; i < num_iterations ; i ++ ) {

        err = clblasSgemv( clblasRowMajor, clblasNoTrans, num_rows - off, num_cols - off, 1,
                           mat_in_A(), 0, lda, vec_in_X(), 0, incx, 1,
                           vec_out_Y(), 0, incy, 1, &command_queue(), 0, NULL, &event );

//    err = clblasSgemv(order, transA, M - off, N - off, alpha,
//                      bufA, offA, lda, bufX, offX, incx, beta,
//                      bufY, offY, incy, 1, &queue, 0, NULL, &event);

        err = clWaitForEvents( 1, &event );

    }

    command_queue.enqueueReadBuffer( vec_out_Y, CL_TRUE, 0, num_rows * 1 * sizeof( float ), vec_Y.data() );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;
    \

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << vec_Y.squaredNorm() << std::endl;

    return time_taken;

}

double MatVectProdTest::CLMatMulBiased( uint num_rows, uint num_cols, uint num_iterations ) {

    std::cout << "Testing matrix product using clBLAS." << std::endl;

    mat_A = Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic >::Random( num_rows, num_cols );
    vec_Y = Eigen::Matrix< float, Eigen::Dynamic, 1 >::Random( num_cols, 1 );
    vec_Y.setZero( num_rows, 1 );


    off  = 1;
    offA = num_rows + off;   /* M + off */
    offX = off;       /* off */
    offY = off;       /* off */

    incx = 1;
    incy = 1;
    lda = num_cols;        /* i.e. lda = N */

    mat_in_A = cl::Buffer( context,
                           CL_MEM_READ_ONLY,
                           num_rows * num_cols * sizeof( float ),
                           NULL, &err );

    vec_in_X = cl::Buffer ( context,
                            CL_MEM_READ_ONLY,
                            num_cols * 1 * sizeof( float ),
                            NULL, &err );

    vec_out_Y = cl::Buffer( context,
                            CL_MEM_READ_WRITE,
                            num_rows * 1 * sizeof( float ),
                            NULL, &err );

    command_queue.enqueueWriteBuffer( mat_in_A, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_A.data() );
    command_queue.enqueueWriteBuffer( vec_in_X, CL_TRUE, 0, num_cols * 1 * sizeof( float ), vec_X.data() );
    command_queue.enqueueWriteBuffer( vec_out_Y, CL_TRUE, 0, num_rows * 1 * sizeof( float ), vec_Y.data() );

    auto start = std::chrono::high_resolution_clock::now();

    for( uint i = 0; i < num_iterations ; i ++ ) {

        err = clblasSgemv( clblasRowMajor, clblasNoTrans, num_rows - off, num_cols - off, 1,
                           mat_in_A(), offA, lda, vec_in_X(), 0, incx, 1,
                           vec_out_Y(), offY, incy, 1, &command_queue(), 0, NULL, &event );

//    err = clblasSgemv(order, transA, M - off, N - off, alpha,
//                      bufA, offA, lda, bufX, offX, incx, beta,
//                      bufY, offY, incy, 1, &queue, 0, NULL, &event);

        err = clWaitForEvents( 1, &event );

    }

    auto end = std::chrono::high_resolution_clock::now();

    command_queue.enqueueReadBuffer( vec_out_Y, CL_TRUE, 0, num_rows * 1 * sizeof( float ), vec_Y.data() );

    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;
    \

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << vec_Y.squaredNorm() << std::endl;

    return time_taken;

}

std::pair< double, double > MatVectProdTest::Run(uint num_rows, uint num_cols , uint num_iterations) {

    std::cout << "Matrix size of " << num_rows << " x " << num_cols << std::endl;

    double time_cpu = CPUMatMul( num_rows, num_cols, num_iterations );
    double time_gpu = CLMatMul( num_rows, num_cols, num_iterations );

    return std::pair< double, double >( time_cpu, time_gpu );

}

std::pair< double, double > MatVectProdTest::RunBiased(uint num_rows, uint num_cols , uint num_iterations) {

    std::cout << "Matrix size of " << num_rows << " x " << num_cols << std::endl;

    double time_cpu = CPUMatMul( num_rows, num_cols, num_iterations );
    double time_gpu = CLMatMulBiased( num_rows, num_cols, num_iterations );

    return std::pair< double, double >( time_cpu, time_gpu );

}
