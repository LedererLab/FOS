#ifndef X_FOS_H
#define X_FOS_H

// C System-Headers
//
// C++ System headers
//#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//
// Project Specific Headers
#include "../Generic/generics.hpp"
#include "../Solvers/solver.hpp"
#include "../Solvers/SubGradientDescent/ISTA/ista.hpp"
#include "../Solvers/SubGradientDescent/FISTA/fista.hpp"
#include "../Solvers/CoordinateDescent/coordinate_descent.hpp"

namespace hdim {

enum class SolverType { ista, fista, cd };

namespace experimental {

template < typename T >
/*!
 * \brief The FOS algorithim
 */
class X_FOS {

//    using Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;
//    using Eigen::Matrix< T, Eigen::Dynamic, 1 > = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

  public:

    X_FOS();
    ~X_FOS();

    /*!
     * \brief Run the main X_FOS algorithm
     *
     * Calling this function will run the X_FOS algorithm using the values of
     * X and Y.
     *
     */
    void operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y, SolverType s_type = SolverType::ista );

    T ReturnLambda();
    T ReturnIntercept();
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
    unsigned int ReturnOptimIndex();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
    Eigen::Matrix< int, Eigen::Dynamic, 1 > ReturnSupport();

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > fos_fit;
    T lambda;
    unsigned int optim_index;

  private:

    Eigen::Matrix< T, Eigen::Dynamic, 1 > X_weights( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X );
    T Y_weight( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > RescaleCoefficients( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& raw_coefs,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x_weights,
            T y_weight);

    T compute_intercept(const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    std::vector< T > GenerateLambdaGrid (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        unsigned int M );

    bool ComputeStatsCond(T C,
                          unsigned int stats_it,
                          T r_stats_it,
                          const std::vector<T> &lambdas,
                          const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas );

    T duality_gap_target( T gamma, T C, T r_stats_it, unsigned int n );

    T primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
                        T r_stats_it );

    T dual_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                      const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                      const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
                      T r_stats_it );

    internal::Solver<T>* solver;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_std_devs;

    T y_std_dev = 0;
    T intercept = 0;

    const T C = 0.75;
    const unsigned int M = 100;
    const T gamma = 1;

    bool statsCont = true;

    unsigned int loop_index = 0;

    unsigned int statsIt = 1;
    std::vector< T > lambda_grid;

    const T L_0 = 0.1;

    int n = 1, p = 1;

};

template< typename T >
/*!
 * \brief Initialize a new algorithm, and instantiate member attributes X and Y.
 *
 * \param x
 * An n x p design matrix
 *
 * \param y
 * An n x 1 vector
 */
X_FOS< T >::X_FOS() {
    static_assert(std::is_floating_point< T >::value, "X_FOS can only be used with floating point types.");
}

template < typename T >
X_FOS<T>::~X_FOS() {
    delete solver;
}

template < typename T >
T X_FOS< T >::ReturnLambda() {
    return lambda;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X_FOS< T >::ReturnBetas() {
    return Betas;
}

template < typename T >
unsigned int X_FOS< T >::ReturnOptimIndex() {
    return optim_index;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS< T >::ReturnCoefficients() {
    return RescaleCoefficients(fos_fit, x_std_devs, y_std_dev );
}

template < typename T >
Eigen::Matrix<int, Eigen::Dynamic, 1> X_FOS<T>::ReturnSupport() {

    T n_t = static_cast<T>( n );
    T cut_off = static_cast<T>( 6 )*C*lambda/n_t;
    DEBUG_PRINT( "Cut-off for Support Computation: " << cut_off );

    return GenerateSupport( fos_fit, cut_off );
//    return fos_fit.unaryExpr( SupportSift<T>( C_t, lambda, n_t ) );

}

template < typename T >
T X_FOS<T>::compute_intercept( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x,
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > scaled_beta = RescaleCoefficients(Beta, x_std_devs, y_std_dev );

    T intercept_part = 0.0;

    for( unsigned int i = 0; i < x.cols() ; i++ ) {

        T X_i_bar = x.col( i ).mean();
        intercept_part += scaled_beta( i )*X_i_bar;

    }

    return y.mean() - intercept_part;
}

template < typename T >
T X_FOS<T>::ReturnIntercept() {
    return intercept;
}

//Member functions

template < typename T >
/*!
 * \brief Determine the 'stop' condition for the outer loop
 *
 * \param stats_it
 * \param r_stats_it
 * \param lambdas
 * \return True if outer loop should continue, false otherwise
 */
bool X_FOS<T>::ComputeStatsCond( T C,
                                 unsigned int stats_it,
                                 T r_stats_it,
                                 const std::vector <T>& lambdas,
                                 const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas ) {

    bool stats_cond = true;

    for ( unsigned int i = 1; i <= stats_it; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_k = Betas.col( i - 1 );
        T rk = lambdas.at( i - 1 );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = Betas.col( stats_it - 1 ) - beta_k;
        T abs_max_betas = beta_diff.template lpNorm< Eigen::Infinity >();

        T n_t = static_cast<T>( n );
        T check_parameter = n_t*abs_max_betas / ( r_stats_it + rk );

        stats_cond &= ( check_parameter <= C );
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
std::vector< T > X_FOS<T>::GenerateLambdaGrid (

    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    unsigned int M ) {

    T rMax = 2.0*( X.transpose() * Y ).template lpNorm< Eigen::Infinity >();
    T rMin = 0.001*rMax;

    return LogScaleVector( rMax, rMin, M );

}

template < typename T >
T X_FOS< T >::primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                                T r_stats_it ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = Y - X*Beta;
    T f_beta = error.squaredNorm() + r_stats_it*Beta.template lpNorm < 1 >();

    return f_beta;

}

template < typename T >
T X_FOS< T >::dual_objective ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                               T r_stats_it ) {

    //Computation of s
    T s_chunk =  r_stats_it / ( 2.0*X.transpose()*( X*Beta - Y ) ).template lpNorm< Eigen::Infinity >();
    T s_chunk_prime = ( - static_cast<T>( Y.transpose()*( X*Beta - Y ) ) )/( Y - X*Beta ).squaredNorm();
    T s = std::min( std::max( - s_chunk, s_chunk_prime ), s_chunk );

    //Computation of nu tilde
    Eigen::Matrix< T, Eigen::Dynamic, 1 > nu_tilde = 2.0*s/r_stats_it*( X*Beta - Y );

    T d_nu = square( r_stats_it )/4.0*( nu_tilde + 2.0/r_stats_it*Y ).squaredNorm() - Y.squaredNorm();

    return d_nu;
}

template < typename T >
T X_FOS<T>::duality_gap_target( T gamma, T C, T r_stats_it, unsigned int n ) {

    T n_f = static_cast<T>( n );
    return gamma*square( C )*square( r_stats_it )/n_f;

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS< T >::X_weights( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > weights( X.cols() );

    for( unsigned int i = 0; i < X.cols() ; i ++ ) {
        Eigen::Matrix< T, Eigen::Dynamic, 1 > X_i = X.col( i );
        weights( i ) = StdDev( X_i );
    }

    return weights;
}

template < typename T >
T X_FOS< T >::Y_weight( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y ) {
    return StdDev( Y );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > X_FOS< T >::RescaleCoefficients(
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& raw_coefs,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x_weights,
    T y_weight ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > scaled_coefs( raw_coefs.size() );

    for( unsigned int i = 0; i < raw_coefs.size() ; i++ ) {

        T weight = y_weight / x_weights( i );
        scaled_coefs( i ) = weight*raw_coefs( i );

    }

    return scaled_coefs;

}

template < typename T >
void X_FOS< T >::operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x, const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y, SolverType s_type ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > old_Betas;

    bool statsCont = true;

    x_std_devs = X_weights( x );
    y_std_dev = Y_weight( y );

    Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic > X = Normalize( x );
    Eigen::Matrix< T , Eigen::Dynamic, 1 > Y = Normalize( y );

    n = X.rows();
    p = X.cols();

    lambda_grid = GenerateLambdaGrid( X, Y, M );

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    switch( s_type ) {
    case SolverType::ista:
        solver = new ISTA<T>();
        break;
    case SolverType::fista:
        solver = new FISTA<T>();
        break;
    case SolverType::cd:
        solver = new CoordinateDescentSolver<T>( X, Y, Betas.col( 0 ) );
        break;
    }

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lambda_grid.at( statsIt - 1 );

        T gap = duality_gap( X, Y, old_Betas, rStatsIt );

        T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

        if( gap <= gap_target ) {

            Betas.col( statsIt - 1 ) = old_Betas;

        } else {

            DEBUG_PRINT( "Current Lambda: " << rStatsIt );

            Betas.col( statsIt - 1 ) = solver->operator()( X, Y, old_Betas, rStatsIt, gap_target );
        }

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lambda_grid, Betas );
    }

    fos_fit = Betas.col( statsIt - 2 );
    lambda = lambda_grid.at( statsIt - 2 );
    optim_index = statsIt;
    intercept = compute_intercept( x, y, fos_fit );

}

}

}

#endif // X_FOS_H
