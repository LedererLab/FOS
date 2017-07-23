#ifndef FOS_JS_HPP
#define FOS_JS_HPP

// C System-Headers
//
// C++ System headers
#include <vector>
// Eigen Headers
//
// Boost Headers
//
// Project Specific Headers
#include "../FOS/x_fos.hpp"

template < typename T >
class JS_FOS : public hdim::X_FOS<T> {

  public:

    /*!
     * \brief Run the main JS_FOS algorithm
     *
     * Calling this function will run the JS_FOS algorithm using the values of
     * X and Y.
     *
     */
    void operator()( std::vector<T>& X_vectorized,
                     std::vector<T>& Y,
                     std::string solver_type );

    T ReturnLambda();
    T ReturnIntercept();
    unsigned int ReturnOptimIndex();
    std::vector<T> ReturnCoefficients();
    std::vector<int> ReturnSupport();

};

template < typename T >
void JS_FOS< T >::operator()( std::vector<T>& X_vectorized,
                              std::vector<T>& Y,
                              std::string solver_type ) {

    (void)solver_type;

    unsigned int n = Y.size();
    unsigned int p = X_vectorized.size() / n ;


    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_eigen = Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >( X_vectorized.data(), n, p );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y_eigen = Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 1 > >( Y.data(), Y.size() );

    return hdim::X_FOS<T>::operator ()( X_eigen,
                                        Y_eigen,
                                        hdim::SolverType::cd );

}


template < typename T >
T JS_FOS< T >::ReturnLambda() {
    return hdim::X_FOS<T>::lambda;
}

template < typename T >
unsigned int JS_FOS< T >::ReturnOptimIndex() {
    return hdim::X_FOS<T>::optim_index;
}

template < typename T >
T JS_FOS< T >::ReturnIntercept() {
    return hdim::X_FOS<T>::intercept;
}

template < typename T >
std::vector<T> JS_FOS< T >::ReturnCoefficients() {

    Eigen::Matrix<T, Eigen::Dynamic, 1 > fos_coefs = hdim::X_FOS<T>::ReturnCoefficients();
    return std::vector<T> (fos_coefs.data(), fos_coefs.data() + fos_coefs.rows() * fos_coefs.cols());

}

template < typename T >
std::vector<int> JS_FOS<T>::ReturnSupport() {

    Eigen::Matrix<int, Eigen::Dynamic, 1 > fos_support = hdim::X_FOS<T>::ReturnSupport();
    return std::vector<int> (fos_support.data(), fos_support.data() + fos_support.rows() * fos_support.cols());

}
#endif // FOS_JS_HPP
