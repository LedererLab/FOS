#ifndef X_FOS_H
#define X_FOS_H

// C System-Headers
//
// C++ System headers
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <memory>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Project Specific Headers
// Generic
#include "../Generic/generics.hpp"
// Solvers
#include "../Solvers/base_solver.hpp"
#include "../Solvers/abstractsolver.hpp"
#include "../Solvers/solver.hpp"
#include "../Solvers/screeningsolver.hpp"
#include "../Solvers/SubGradientDescent/ISTA/ista.hpp"
#include "../Solvers/SubGradientDescent/FISTA/fista.hpp"
#include "../Solvers/CoordinateDescent/coordinate_descent.hpp"

namespace hdim {

enum class SolverType { ista, screen_ista, fista, screen_fista, cd, screen_cd };

template < typename T >
/*!
 * \brief The FOS algorithim
 */
class X_FOS {

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
    void operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                     SolverType s_type = SolverType::ista );

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
    T intercept = 0;

  private:

    Eigen::Matrix< T, Eigen::Dynamic, 1 > X_weights( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X );

    T Y_weight( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > RescaleCoefficients( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& raw_coefs,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x_weights,
            T y_weight);

    T compute_intercept(const Eigen::Matrix< T, Eigen::Dynamic, 1 >& x,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& fit );

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

    Eigen::Matrix< int, Eigen::Dynamic, 1 > ApplyScreening( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& nu_tilde,
            const T duality_gap,
            const T lambda );

    void choose_solver( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& beta,
                        SolverType s_type );

    std::unique_ptr< internal::BaseSolver<T> > solver;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > x_std_devs;

    T y_std_dev = 0;

    const T C = 0.75;
    const unsigned int M = 100;
    const T gamma = 1;

    unsigned int loop_index = 0;

    unsigned int statsIt = 1;

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
X_FOS<T>::~X_FOS() {}

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
                               const Eigen::Matrix< T, Eigen::Dynamic, 1 >& fit ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > scaled_beta = RescaleCoefficients( fit, x_std_devs, y_std_dev );

    T intercept_part = 0.0;

    for( unsigned int i = 0; i < x.cols() ; i++ ) {

        T X_i_bar = x.col( i ).mean();
        intercept_part += scaled_beta( i )*X_i_bar;

    }

    return static_cast<T>( y.mean() ) - static_cast<T>( intercept_part );
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

        T weight = ( x_weights(i) == 0.0 )?( 0.0 ):( y_weight / x_weights(i) );
        scaled_coefs( i ) = weight*raw_coefs( i );

    }

    return scaled_coefs;

}

template < typename T >
Eigen::Matrix< int, Eigen::Dynamic, 1 > X_FOS< T >::ApplyScreening( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& nu_tilde,
        const T duality_gap,
        const T lambda ) {

    Eigen::Matrix< int, Eigen::Dynamic, 1 > A( x.cols() );
    unsigned int counter = 0;

    for( unsigned int j = 0; j < x.cols(); j ++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > X_j = x.col( j );

        T radius = std::abs( static_cast<T>( X_j.transpose() * nu_tilde ) ) + std::sqrt( 2.0 / square( lambda ) * duality_gap )*X_j.norm();
        A[j] = ( radius > 1.0 );

        if ( radius > 1.0 ) {
            counter ++;
        }

    }

    DEBUG_PRINT( "Number of active variables: " << counter );

    return A;

}

template < typename T >
void X_FOS< T >::choose_solver( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                                const Eigen::Matrix< T, Eigen::Dynamic, 1 >& beta,
                                SolverType s_type ) {

    solver.reset();

    switch( s_type ) {
    case SolverType::ista:
        solver = std::unique_ptr< ISTA<T,internal::Solver<T>> >( new ISTA<T,internal::Solver<T>>() );
        break;
    case SolverType::screen_ista:
        solver = std::unique_ptr< ISTA<T,internal::ScreeningSolver<T>> >( new ISTA<T,internal::ScreeningSolver<T>>() );
        break;
    case SolverType::fista:
        solver = std::unique_ptr< FISTA<T,internal::Solver<T>> >( new FISTA<T,internal::Solver<T>>(beta) );
        break;
    case SolverType::screen_fista:
        solver = std::unique_ptr< FISTA<T,internal::ScreeningSolver<T>> >( new FISTA<T,internal::ScreeningSolver<T>>(beta) );
        break;
    case SolverType::cd:
        solver = std::unique_ptr< LazyCoordinateDescent<T,internal::Solver<T>> >( new LazyCoordinateDescent<T,internal::Solver<T>>( x, y, Betas.col( 0 ) ) );
        break;
    case SolverType::screen_cd:
        solver = std::unique_ptr< LazyCoordinateDescent<T,internal::ScreeningSolver<T>> >( new LazyCoordinateDescent<T,internal::ScreeningSolver<T>>( x, y, Betas.col( 0 ) ) );
        break;
    }

}

template < typename T >
void X_FOS< T >::operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
                             const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
                             SolverType s_type ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > old_Betas;

    bool statsCont = true;

    x_std_devs = X_weights( x );
    y_std_dev = Y_weight( y );

    Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic > X = Normalize( x );
    Eigen::Matrix< T , Eigen::Dynamic, 1 > Y = Normalize( y );

    n = X.rows();
    p = X.cols();

    std::vector<T> lambda_grid = GenerateLambdaGrid( X, Y, M );

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );

    choose_solver( X, Y, Betas.col( 0 ), s_type );

    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lambda_grid.at( statsIt - 1 );

        T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

        DEBUG_PRINT( "Current Lambda: " << rStatsIt );
        Betas.col( statsIt - 1 ) = solver->operator()( X, Y, old_Betas, rStatsIt, gap_target );

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lambda_grid, Betas );
    }

    fos_fit = Betas.col( statsIt - 2 );
    lambda = lambda_grid.at( statsIt - 2 );
    optim_index = statsIt;
//    intercept = compute_intercept( x, y, fos_fit );

    // Computation of Intercept -- function call causes exception when compilied with -DDEBUG flag
    // No idea why this is happening -- does not seem to be an issue when compilied using release parameters

    Eigen::Matrix< T, Eigen::Dynamic, 1 > scaled_beta = RescaleCoefficients( fos_fit, x_std_devs, y_std_dev );

    T intercept_part = 0.0;
    for( unsigned int i = 0; i < x.cols() ; i++ ) {
        T X_i_bar = x.col( i ).mean();
        intercept_part += scaled_beta( i )*X_i_bar;
    }

    intercept = y.mean() - intercept_part;

}

}

#endif // X_FOS_H
