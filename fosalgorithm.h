#ifndef FOSALGORITHM_H
#define FOSALGORITHM_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// SPAMS Headers
#include "linalg.h" // AbstractMatrix and Matrix
#include "spams.h"
// Project Specific Headers
//

template < typename T, uint m, uint n >
Eigen::Matrix< T, m, n > Spams2EigenMat ( const Matrix<T>* spams_mat ) {
    auto M = Eigen::Map< Eigen::Matrix< T, n, m, Eigen::ColMajor> >( spams_mat->rawX() );
    return M;
}

template < typename T, uint m, uint n >
Eigen::Matrix< T, m, n > Spams2EigenMat ( Matrix<T>* spams_mat ) {

    auto M = Eigen::Map< Eigen::Matrix< T, n, m, Eigen::ColMajor> >( spams_mat->rawX() );
    return M;
}

template < typename T, uint m, uint n >
Matrix<T>* Eigen2SpamsMat ( const Eigen::Matrix< T, n, m >& eigen_mat ) {

    auto eigen_mat_size = eigen_mat.cols() * eigen_mat.rows();

    T* non_const_mat_data = new T[eigen_mat_size]; // create a new buffer
    auto mat_data = eigen_mat.data();

    std::copy( mat_data, mat_data + eigen_mat_size, non_const_mat_data); // copy the data

    auto spams_mat = new Matrix<T> ( non_const_mat_data, m, n );

    return spams_mat;
}

template < typename T, uint m, uint n >
AbstractMatrixB<T> Eigen2SpamsAbstractMatB ( const Eigen::Matrix< T, n, m >& eigen_mat ) {
    return AbstractMatrixB<T>( eigen_mat.data(), m, n );
}


//spams.fistaFlat( y,x,mat_W0,FALSE,numThreads = 8, ista = TRUE, verbose = FALSE, max_it= 1,L0 = 0.1, loss = 'square',regul = 'l1', lambda1 = 0.5*rStatsIt)
/*Matrix<T> *_fistaFlat(Matrix<T> *X,
                      AbstractMatrixB<T> *D,
                      Matrix<T> *alpha0,
                      Matrix<T> *alpha,
                      Vector<int> *groups, // params
                      int num_threads,
                      int max_it,
                      T L0,
                      bool fixed_step,
                      T gamma,
                      T _lambda,
                      T delta,
                      T lambda2,
                      T lambda3,
                      T a,
                      T b,
                      T c,
                      T tol,
                      int it0,
                      int max_iter_backtracking,
                      bool compute_gram,
                      bool lin_admm,
                      bool admm,
                      bool intercept,
                      bool resetflow,
                      char* name_regul,
                      char* name_loss,
                      bool verbose,
                      bool pos,
                      bool clever,
                      bool log,
                      bool ista,
                      bool subgrad,
                      char* logName,
                      bool is_inner_weights,
                      Vector<T> *inner_weights,
                      int size_group,
                      bool sqrt_step,
                      bool transpose,
                      int linesearch_mode
                     )
                     */

/*fistaFlat(Y,
          X,
          W0,
          return_optim_info = FALSE,
          numThreads =-1,
          max_it =1000,
          L0=1.0,
          fixed_step=FALSE,
          gamma=1.5,
          lambda1=1.0,
          delta=1.0,
          lambda2=0.,
          lambda3=0.,
          a=1.0,
          b=0.,
          c=1.0,
          tol=0.000001,
          it0=100,
          max_iter_backtracking=1000,
          compute_gram=FALSE,
          lin_admm=FALSE,
          admm=FALSE,
          intercept=FALSE,
          resetflow=FALSE,
          regul="",
          loss="",
          verbose=FALSE,
          pos=FALSE,
          clever=FALSE,
          log=FALSE,
          ista=FALSE,
          subgrad=FALSE,
          logName="",
          is_inner_weights=FALSE,
          inner_weights=c(0.),
          size_group=1,
          groups = NULL,
          sqrt_step=TRUE,
          transpose=FALSE,
          linesearch_mode=0)
          */

//lambda_1 = 0.5*rStatsIt

//Will need to be deleted by user
char* str_to_c_ptr( std::string& str ) {

    char * writable = new char[str.size() + 1];
    std::copy(str.begin(), str.end(), writable);
    writable[str.size()] = '\0'; // don't forget the terminating 0

    return writable;

}

template < typename T, uint m, uint n >
Matrix<T>* FistaFlat( Matrix<T>* Y, Matrix<T>* X, Matrix<T>* Omega_0, const T lambda_1 ) {

    //Initialize alpha
    auto W = new Matrix<T> ( m, n );
    W->setZeros();

    //Initialize groups
    auto groups = new Vector<int>( n );
    groups->setZeros();

    //Initialize num_threads
    int num_threads = omp_get_max_threads();

    auto inner_weights = new Vector<T>( 0 );

//    std::string regul = "l1";
//    char* regul_ptr = str_to_c_ptr( regul );

//    std::string loss = "square";
//    char* loss_ptr = str_to_c_ptr( loss );

//    std::string log_name = "";
//    char* log_name_ptr = str_to_c_ptr( log_name );

    char regul[] = "l1";
    char loss[] = "square";
    char log_name[] = "";

    auto ret_val =_fistaFlat(Y, //X
                             X, //D
                             Omega_0, //alpha0
                             W, // alpha
                             groups, // groups
                             num_threads, // num_threads
                             1, // mat_it
                             0.1f, //L0
                             false, //fixed_step
                             1.5f, // gamma
                             lambda_1, //lambda_
                             1.0f, //delta
                             0.0f, //lambda2
                             0.0f, //lambda3
                             1.0f, //a
                             0.0f, //b
                             1.0f, //c
                             0.000001f, //tol
                             100, //it0
                             1000, //max_iter_backtracking
                             false, //compute_gram
                             false, //lin_admm
                             false, //admm
                             false, //intercept
                             false, //resetflow
                             regul, //name_regul
                             loss, //name_loss
                             false, //verbose
                             false, //pos
                             false, //clever
                             false, //log
                             true, //ista
                             false, //subgrad
                             log_name, //logName
                             false, //is_inner_weights
                             inner_weights, //inner_weights
                             1, //size_group
                             true, //sqrt_step
                             false, //transpose
                             0 //linesearch_mode
                            );

    return ret_val;
}

#endif // FOSALGORITHM_H
