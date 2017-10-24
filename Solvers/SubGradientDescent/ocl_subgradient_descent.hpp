#ifndef OCL_SUBGRADIENT_DESCENT_HPP
#define OCL_SUBGRADIENT_DESCENT_HPP

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// Boost Headers
//
// OpenCL Headers
#include <clBLAS.h>
// Project Specific Headers
#include "../../Generic/generics.hpp"
#include "../../Generic/ocl_debug.hpp"
#include "../ocl_solver.hpp"

namespace hdim {

namespace ocl {

namespace internal {

template < typename T >

/*!
 * \brief Abstract base class for Sub-Gradient Descent algorithms
 * ,such as ISTA and FISTA, with backtracking line search.
 */
class SubGradientSolver : public ocl::internal::Solver<T> {

  public:
    SubGradientSolver( T L = 0.1 );
    ~SubGradientSolver();

  protected:

    void f_beta (
        const cl::Buffer& X,
        const cl::Buffer& Y,
        const cl::Buffer& Beta );

    void f_beta_tilda (
        const cl::Buffer& X,
        const cl::Buffer& Y,
        const cl::Buffer& Beta,
        const cl::Buffer& Beta_prime,
        T L );

    void update_beta_ista (
        const cl::Buffer& X,
        const cl::Buffer& Y,
        const cl::Buffer& Beta,
        T L,
        T thres );

    const T L_0;

  private:

    void OCLInit( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                  const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                  const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    cl::Buffer  Beta_, X_, Y_; // Input argument buffers
    cl::Buffer ScratchVector_, ScratchScalar_, ScratchDoubleVector_; // Scratch buffers

    T ScratchScalar;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ScratchVector; // Host copy of Beta

    std::vector< cl::Buffer > X_cols_; // Buffers containing columns of design matrix

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta; // Host copy of Beta

    cl_int err;
    cl_event event = NULL;

    size_t off;
    size_t offMat, offVec;

    int incx = 1;
    int incy = 1;
    size_t ldMat;

    int n, p; // rows and columns of X respectively

    std::string softthreshold_kernel = R"END(

       __kernel void SoftThreshold( __global TYPE* input, __global TYPE* output, const TYPE threshold, const int n, const int p )
       {

           int i = get_global_id(0);

           TYPE X_i_j = input[i];
           TYPE signum = (TYPE)( X_i_j >= 0.0 );

           TYPE fragment = fabsf( X_i_j ) - threshold;
           TYPE pos_part = ( fragment >= 0.0 )?( fragment ):( 0.0 );

           output[i] = signum*pos_part;

       }

  )END";

};

template < typename T >
SubGradientSolver< T >::SubGradientSolver( T L ) : L_0( L ) {
    static_assert(std::is_floating_point< T >::value,\
                  "Subgradient descent methods can only be used with floating point types.");
}

template < typename T, typename Base >
SubGradientSolver< T, Base >::~SubGradientSolver() {}

template < typename T >
SubGradientSolver< T >::OCLInit( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                                 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    n = X.rows();
    p = Y.cols();

    X_ = cl::Buffer ( OpenCLBase::context,
                      CL_MEM_READ_ONLY,
                      n * p * sizeof( T ),
                      NULL, &err );

    OCL_DEBUG( err );

    Beta_ = cl::Buffer ( OpenCLBase::context,
                         CL_MEM_READ_ONLY,
                         p * 1 * sizeof( T ),
                         NULL, &err );

    OCL_DEBUG( err );

    Y_ = cl::Buffer ( OpenCLBase::context,
                      CL_MEM_READ_ONLY,
                      n * 1 * sizeof( T ),
                      NULL, &err );

    OCL_DEBUG( err );

    ScratchScalar_ = cl::Buffer( OpenCLBase::context,
                                 CL_MEM_READ_WRITE,
                                 1 * 1 * sizeof( T ),
                                 NULL, &err );

    OCL_DEBUG( err );

    ScratchVector_ = cl::Buffer( OpenCLBase::context,
                                 CL_MEM_READ_WRITE,
                                 n * 1 * sizeof( T ),
                                 NULL, &err );

    OCL_DEBUG( err );

    ScratchDoubleVector_ = cl::Buffer( OpenCLBase::context,
                                       CL_MEM_READ_WRITE,
                                       2 * n * 1 * sizeof( T ),
                                       NULL, &err );

    OCL_DEBUG( err );

    off  = 1;
    offMat = n + off;   /* M + off */
    offVec = off;       /* off */

    incx = 1;
    incy = 1;
    ldMat = p;        /* i.e. lda = N */

    ocl::internal::Solver<T>::command_queue.enqueueWriteBuffer( X_, CL_TRUE, 0, n * p * sizeof( T ), X.data() );
    ocl::internal::Solver<T>::command_queue.enqueueWriteBuffer( Y_, CL_TRUE, 0, n * 1 * sizeof( T ), Y.data() );
    ocl::internal::Solver<T>::command_queue.enqueueWriteBuffer( Beta_, CL_TRUE, 0, n * 1 * sizeof( T ), Beta.data() );
}

template < typename T >
void SubGradientSolver< T, Base >::f_beta (
    const cl::Buffer& X,
    const cl::Buffer& Y,
    const cl::Buffer& Beta ) {

    // Since BLAS Matrix-Vector product would over-write Beta we copy
    // it into out scratch buffer for vector objects
    err = clEnqueueCopyBuffer( &ocl::internal::Solver<T>::command_queue(),
                               Beta(),
                               ScratchVector_(),
                               off,
                               off,
                               n * 1 * sizeof( T ),
                               0,
                               NULL,
                               &event );

    OCL_DEBUG( err );

    // Compute X*Beta - Y and load results into vector-sized scratch buffer
    err = clblasSgemv( clblasRowMajor,
                       clblasNoTrans,
                       n,
                       p,
                       1.0,
                       X(),
                       offMat,
                       ldMat,
                       Y(),
                       offVec,
                       incy,
                       -1.0,
                       ScratchVector_(),
                       offVec,
                       incy,
                       1,
                       &ocl::internal::Solver<T>::command_queue(),
                       0,
                       NULL,
                       &event );


    OCL_DEBUG( err );


    // Computer ( X*Beta - Y )_2^2 and load results into scalar-sized scratch buffer
    err = clblasSnrm2( n * p,
                       ScratchScalar_(),
                       offVec,
                       ScratchVector_(),
                       offMat,
                       incx,
                       ScratchDoubleVector_(),
                       1,
                       &ocl::internal::Solver<T>::command_queue(),
                       0,
                       NULL,
                       &event );

    OCL_DEBUG( err );

//    return (X*Beta - Y).squaredNorm();

}

template < typename T >
void SubGradientSolver< T, Base >::f_beta_tilda (
    const cl::Buffer& X,
    const cl::Buffer& Y,
    const cl::Buffer& Beta,
    const cl::Buffer& Beta_prime,
    T L ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_beta = X*Beta_prime - Y;
    T taylor_term_0 = f_beta.squaredNorm();

    Eigen::Matrix< T, Eigen::Dynamic, 1  > f_grad = 2.0*X.transpose()*( f_beta );
    Eigen::Matrix< T, Eigen::Dynamic, 1  > beta_diff = ( Beta - Beta_prime );

    T taylor_term_1 = f_grad.transpose()*beta_diff;

    T taylor_term_2 = L/2.0*beta_diff.squaredNorm();

    return taylor_term_0 + taylor_term_1 + taylor_term_2;
}

template < typename T >
void SubGradientSolver< T, Base >::update_beta_ista (
    const cl::Buffer& X,
    const cl::Buffer& Y,
    const cl::Buffer& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

}

}

}

#endif // OCL_SUBGRADIENT_DESCENT_HPP
