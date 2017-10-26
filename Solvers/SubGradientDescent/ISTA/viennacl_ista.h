#ifndef VIENNACL_ISTA_H
#define VIENNACL_ISTA_H

// C System-Headers
//
// C++ System headers
#include <functional>
// Eigen Headers
#define VIENNACL_WITH_EIGEN 1
// Boost Headers
//
// ViennCL Headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
// OpenMP Headers
//
// Project Specific Headers
#include "../../../Generic/generics.hpp"
#include "../../../Generic/debug.hpp"
#include "../viennacl_subgradient_descent.hpp"

namespace hdim {

namespace vcl {

template < typename T, typename Base = vcl::internal::Solver< T > >
/*!
 * \brief Run the Iterative Shrinking and Thresholding Algorthim.
 */
class ISTA : public vcl::internal::SubGradientSolver<T,Base> {

  public:
    ISTA( T L_0 = 0.1 );

  protected:
    viennacl::vector<T> update_rule(
        const viennacl::matrix<T>& X,
        const viennacl::vector<T>& Y,
        const viennacl::vector<T>& Beta_0,
        T lambda );

  private:
    const T eta = 1.5;
    T L = static_cast<T>( 0 );

};

template < typename T, typename Base >
ISTA<T,Base>::ISTA( T L_0 ) : internal::SubGradientSolver<T,Base>( L_0 ) {}

template < typename T, typename Base >
viennacl::vector<T> ISTA<T,Base>::update_rule(
    const viennacl::matrix<T>& X,
    const viennacl::vector<T>& Y,
    const viennacl::vector<T>& Beta_0,
    T lambda ) {

    viennacl::vector<T> Beta = Beta_0;

    unsigned int counter = 0;
    L = internal::SubGradientSolver<T,Base>::L_0;

    viennacl::vector<T> Beta_temp = internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    counter++;
    DEBUG_PRINT( "Backtrace iteration: " << counter );

    while( ( internal::SubGradientSolver<T,Base>::f_beta( X, Y, Beta_temp ) > internal::SubGradientSolver<T,Base>::f_beta_tilda( X, Y, Beta_temp, Beta, L ) ) ) {

        counter++;
        DEBUG_PRINT( "Backtrace iteration: " << counter );

        L*= eta;
        Beta_temp = internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );

    }

    return internal::SubGradientSolver<T,Base>::update_beta_ista( X, Y, Beta, L, lambda );
}

}

}

#endif // VIENNACL_ISTA_H
