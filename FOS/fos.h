#ifndef FOS_H
#define FOS_H

// C System-Headers
//
// C++ System headers
#include <vector>
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//
// Project Specific Headers
//#include "../Generic/algorithm.h"
#include "../Generic/debug.h"
#include "../Generic/generics.h"
//#include "../ISTA/ista.h"

namespace hdim {

template < typename T >
/*!
 * \brief The main FOS algorithim
 */
class FOS {

  public:
    FOS( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x, Eigen::Matrix< T, Eigen::Dynamic, 1 > y );
    void Algorithm();

    T ReturnLambda();
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
    uint ReturnOptimIndex();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
    Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnSupport();

  protected:
    Eigen::Matrix< T, Eigen::Dynamic, 1 > avfos_fit;
    T lambda;
    uint optim_index;

  private:
    std::vector< T > GenerateLambdaGrid(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X,
                                        const Eigen::Matrix<T, Eigen::Dynamic, 1 > &Y,
                                        uint M);


    bool ComputeStatsCond(
        T C,
        uint stats_it,
        T r_stats_it,
        const std::vector<T>& lambdas,
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas );

    T DualityGap( uint r_stats_it );

    T duality_gap_target( T gamma, T C, T r_stats_it, uint n );
    T dual_objective ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                       const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                       const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                       T r_stats_it );
    T primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                        T r_stats_it );

    T f_beta (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta );

    T f_beta_tilda (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
        T L );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > update_beta_ista (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
        T L,
        T thres );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ISTA (
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, \
        uint num_iterations, \
        T L_0, \
        T lambda );

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Betas;
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > old_Betas;

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > X;
    Eigen::Matrix< T, Eigen::Dynamic, 1 > Y;

    const T C = 0.75;
    const uint M = 100;
    const T gamma = 1;

    T rMax;
    T rMin;

    T L_k_less_1 = 0.1;
    uint statsIt = 1;

    uint n = 0, p = 0;

    std::vector< T > lambda_grid;

    uint loop_index = 0;

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
FOS< T >::FOS(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x, Eigen::Matrix<T, Eigen::Dynamic, 1 > y ) : X( x ), Y( y ) {}

template < typename T >
T FOS< T >::ReturnLambda() {
    return lambda;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > FOS< T >::ReturnBetas() {
    return Betas;
}

template < typename T >
uint FOS< T >::ReturnOptimIndex() {
    return optim_index;
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > FOS< T >::ReturnCoefficients() {
    return avfos_fit;
}

template < typename T >
Eigen::Matrix<T, Eigen::Dynamic, 1> FOS<T>::ReturnSupport() {

    T C_t = static_cast<T>( C );
    T n_t = static_cast<T>( X.rows() );

    return avfos_fit.unaryExpr( SupportSift<T>( C_t, lambda, n_t ) );

}

//Free functions

template < typename T >
T compute_lp_norm( T& matrix, int norm_type ) {
    return matrix.template lpNorm< norm_type >();
}

template < typename T >
T compute_sqr_norm( T& matrix ) {
    return matrix.squaredNorm();
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
bool FOS<T>::ComputeStatsCond(T C,
                              uint stats_it,
                              T r_stats_it,
                              const std::vector<T> &lambdas,
                              const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                              const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& Betas ) {

    bool stats_cond = true;

    for ( uint i = 1; i <= stats_it; i++ ) {

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_k = Betas.col( i - 1 );
        T rk = lambdas.at( i - 1 );

        Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_diff = Betas.col( stats_it - 1 );
        beta_diff -= beta_k;
        T abs_max_betas = beta_diff.cwiseAbs().maxCoeff();

        T n = static_cast<T>( X.rows() );

        T check_parameter = n*abs_max_betas / ( r_stats_it + rk );

        stats_cond &= ( check_parameter <= C );
    }

    return stats_cond;

}

template < typename T >
/*!
 * \brief Generate the 'rs' vector
 * \return
 */
std::vector< T > FOS<T>::GenerateLambdaGrid( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        uint M ) {

    T rMax = 2.0*( X.transpose() * Y ).template lpNorm< Eigen::Infinity >();
    T rMin = 0.001*rMax;

    return LogScaleVector( rMax, rMin, M );

}

template < typename T >
/*!
 * \brief Compute the target for the duality gap used in the inner loop
 *
 * The duality gap should be less than or equal to this target in order to
 * exit the inner loop.
 *
 * \param r_stats_it
 *
 * \return Target quantity
 */
T FOS< T >::duality_gap_target( T gamma, T C, T r_stats_it, uint n ) {

    T n_f = static_cast<T>( n );
    return gamma*square( C )*square( r_stats_it )/n_f;

}

template < typename T >
T FOS< T >::primal_objective( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
                              const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
                              const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta, \
                              T r_stats_it ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > error = Y - X*Beta;
    T f_beta = error.squaredNorm() + r_stats_it*Beta.template lpNorm < 1 >();

    return f_beta;

}

template < typename T >
T FOS< T >::dual_objective ( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
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
T FOS<T>::f_beta (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1  >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta ) {

    return (X*Beta - Y).squaredNorm();

}

template < typename T >
T FOS<T>::f_beta_tilda (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_prime,
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
Eigen::Matrix< T, Eigen::Dynamic, 1 > FOS<T>::update_beta_ista (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta,
    T L,
    T thres ) {

    Eigen::Matrix< T, Eigen::Dynamic, 1 > f_grad = 2.0*( X.transpose()*( X*Beta - Y ) );
    Eigen::Matrix< T, Eigen::Dynamic, 1 > beta_to_modify = Beta - (1.0/L)*f_grad;

    return beta_to_modify.unaryExpr( SoftThres<T>( thres/L ) );

}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > FOS<T>::ISTA (
    const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y, \
    const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0, \
    uint num_iterations, \
    T L_0, \
    T lambda ) {

    T eta = 1.5;
    T L = L_0;

    Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta = Beta_0;

//    #pragma omp parallel for
    for( uint i = 0; i < num_iterations; i++ ) {

        uint counter = 0;

        Eigen::Matrix< T, Eigen::Dynamic, 1 > Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        counter++;
        //DEBUG_PRINT( "Backtrace iteration: " << counter );

        while( ( f_beta( X, Y, Beta_temp ) > f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

            counter++;
            //DEBUG_PRINT( "Backtrace iteration: " << counter );
            L*= eta;
            Beta_temp = update_beta_ista( X, Y, Beta, L, lambda );

        }

        Beta = update_beta_ista( X, Y, Beta, L, lambda );

    }

    L_k_less_1 = L;

    return Beta;

}

template< typename T >
/*!
 * \brief Run the main FOS algorithm
 *
 * Calling this function will run the FOS algorithm using the values of
 * X and Y that were instantiated with the class constructor.
 *
 */
void FOS< T >::Algorithm() {

    X = Normalize( X );
    Y = Normalize( Y );

    lambda_grid = GenerateLambdaGrid( X, Y, M );

    bool statsCont = true;
    uint statsIt = 1;

    Betas = Eigen::Matrix< T , Eigen::Dynamic, Eigen::Dynamic >::Zero( X.cols(), M );


    //Outer Loop
    while( statsCont && ( statsIt < M ) ) {

        statsIt ++;

        DEBUG_PRINT( "Outer loop #: " << statsIt );

        old_Betas = Betas.col( statsIt - 2 );
        T rStatsIt = lambda_grid.at( statsIt - 1 );

        //Inner Loop
        while( true ) {

            loop_index ++;
            //DEBUG_PRINT( "Inner loop #: " << loop_index );

//            Eigen::Matrix< T , Eigen::Dynamic, 1  > beta_k = Betas.col( statsIt - 1 );
//            T duality_gap = primal_objective( X, Y, beta_k, rStatsIt ) + dual_objective( X, Y, beta_k, rStatsIt );

            T duality_gap = primal_objective( X, Y, old_Betas, rStatsIt ) + dual_objective( X, Y, old_Betas, rStatsIt );


            uint n = static_cast< uint >( X.rows() );
            T gap_target = duality_gap_target( gamma, C, rStatsIt, n );

            DEBUG_PRINT( "Duality gap is " << duality_gap << " gap target is " << gap_target );

            //Criteria meet, exit loop
            if( duality_gap <= gap_target ) {

                //DEBUG_PRINT( "Duality gap is below specified threshold, exiting inner loop." );
                Betas.col( statsIt - 1 ) = old_Betas;
                loop_index = 0;

                L_k_less_1 = 0.1;
                break;

            } else {

//                Betas.col( statsIt - 1 ) = FistaFlat<T>( Y, X, old_Betas, 0.5*rStatsIt );
                Betas.col( statsIt - 1 ) = ISTA( X, Y, old_Betas, 1, L_k_less_1, rStatsIt );

                old_Betas = Betas.col( statsIt - 1 );

            }

        }

        statsCont = ComputeStatsCond( C, statsIt, rStatsIt, lambda_grid, X, Betas );
    }

    avfos_fit = Betas.col( statsIt - 2 );
    lambda = lambda_grid.at( statsIt - 2 );
    optim_index = statsIt;

    std::cout << "Stopping Index: " << optim_index << std::endl;

}

}

//Eigen::Matrix< double, Eigen::Dynamic, 1 > estimate_support_FOS( Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > X,
//                                                                 Eigen::Matrix< double, Eigen::Dynamic, 1 > Y );

#endif // FOS_H
