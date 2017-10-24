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

MatProdTest::MatProdTest( uint platform_number, uint device_number ) : OpenCLBase( platform_number, device_number ) {

    err = clblasSetup();
}

MatProdTest::~MatProdTest() {

    clblasTeardown();

}

double MatProdTest::CPUMatMul( uint num_rows, uint num_cols ) {

    std::cout << "Testing matrix product using Eigen3." << std::endl;

//    mat_X = build_matrix< float >( num_rows, num_cols, &eucl_distance );
//    mat_Y = build_matrix< float >( num_rows, num_cols, &eucl_distance );
    mat_X = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Y = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Z.setZero( num_rows, num_cols );

    auto start = std::chrono::high_resolution_clock::now();

    mat_Z = mat_X*mat_Y;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;\

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << mat_Z.squaredNorm() << std::endl;

    return time_taken;

}

double MatProdTest::CLMatMul( uint num_rows, uint num_cols ) {

    std::cout << "Testing matrix product using clBLAS." << std::endl;

//    mat_X = build_matrix< float >( num_rows, num_cols, &eucl_distance );
//    mat_Y = build_matrix< float >( num_rows, num_cols, &eucl_distance );
    mat_X = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Y = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Z.setZero( num_rows, num_cols );

    ld_X = ld_Y = ld_Z = num_rows;

    mat_in_X = cl::Buffer( context,
                               CL_MEM_READ_ONLY,
                               num_rows * num_cols* sizeof( float ),
                               NULL, &err );

    //cl::Buffer ( context, CL_MEM_READ_WRITE, processed_bytes, err_ptr );
    mat_in_Y = cl::Buffer ( context,
                               CL_MEM_READ_ONLY,
                               num_rows * num_cols * sizeof( float ),
                               NULL, &err );

    mat_out_Z = cl::Buffer( context,
                                CL_MEM_READ_WRITE,
                                num_rows * num_cols * sizeof( float ),
                                NULL, &err );

    auto start = std::chrono::high_resolution_clock::now();

    command_queue.enqueueWriteBuffer( mat_in_X, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_X.data() );
    command_queue.enqueueWriteBuffer( mat_in_Y, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Y.data() );
    command_queue.enqueueWriteBuffer( mat_out_Z, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Z.data() );


    err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                            num_rows, num_rows, num_rows,
                            1, mat_in_X(), 0, ld_X,
                            mat_in_Y(), 0, ld_Y, 1,
                            mat_out_Z(), 0, ld_Z,
                            1, &command_queue(), 0, NULL, &event );

    err = clWaitForEvents( 1, &event );

    command_queue.enqueueReadBuffer( mat_out_Z, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Z.data() );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;\

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << mat_Z.squaredNorm() << std::endl;

    return time_taken;

}

double MatProdTest::CLMatMulBiased( uint num_rows, uint num_cols ) {

    std::cout << "Testing matrix product using clBLAS." << std::endl;

    mat_X = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Y = Eigen::MatrixXf::Random( num_rows, num_cols );
    mat_Z.setZero( num_rows, num_cols );

    ld_X = ld_Y = ld_Z = num_rows;

    mat_in_X = cl::Buffer( context,
                               CL_MEM_READ_ONLY,
                               num_rows * num_cols* sizeof( float ),
                               NULL, &err );

    //cl::Buffer ( context, CL_MEM_READ_WRITE, processed_bytes, err_ptr );
    mat_in_Y = cl::Buffer ( context,
                               CL_MEM_READ_ONLY,
                               num_rows * num_cols * sizeof( float ),
                               NULL, &err );

    mat_out_Z = cl::Buffer( context,
                                CL_MEM_READ_WRITE,
                                num_rows * num_cols * sizeof( float ),
                                NULL, &err );

    command_queue.enqueueWriteBuffer( mat_in_X, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_X.data() );
    command_queue.enqueueWriteBuffer( mat_in_Y, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Y.data() );
    command_queue.enqueueWriteBuffer( mat_out_Z, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Z.data() );

    auto start = std::chrono::high_resolution_clock::now();

    err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                            num_rows, num_rows, num_rows,
                            1, mat_in_X(), 0, ld_X,
                            mat_in_Y(), 0, ld_Y, 1,
                            mat_out_Z(), 0, ld_Z,
                            1, &command_queue(), 0, NULL, &event );

    err = clWaitForEvents( 1, &event );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    auto time_taken = ms.count();
    std::cout<< "Matrix product took " << time_taken <<" ms." << std::endl;\

    command_queue.enqueueReadBuffer( mat_out_Z, CL_TRUE, 0, num_rows * num_cols * sizeof( float ), mat_Z.data() );

    std::cout << "Squared Frobenius norm of result." << std::endl;
    std::cout << mat_Z.squaredNorm() << std::endl;

    return time_taken;

}

std::pair< double, double > MatProdTest::Run( uint num_rows, uint num_cols ) {

    std::cout << "Matrix size of " << num_rows << " x " << num_cols << std::endl;

    double time_cpu = CPUMatMul( num_rows, num_cols );
    double time_gpu = CLMatMul( num_rows, num_cols );

    return std::pair< double, double >( time_cpu, time_gpu );

}

std::pair< double, double > MatProdTest::RunBiased( uint num_rows, uint num_cols ) {

    std::cout << "Matrix size of " << num_rows << " x " << num_cols << std::endl;

    double time_cpu = CPUMatMul( num_rows, num_cols );
    double time_gpu = CLMatMulBiased( num_rows, num_cols );

    return std::pair< double, double >( time_cpu, time_gpu );

}