#ifndef FOS_H
#define FOS_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// FISTA Headers
//#include <spams.h> // _fistaFlat
// Project Specific Headers
//

template < typename T, uint n, uint p >
class FOS {

  public:
    FOS( Eigen::Matrix< T, n, p >& X, Eigen::Matrix< T, n, 1 >& Y );

  private:
    T Mean( Eigen::Matrix< T, n, p >& mat );
    T StdDev( Eigen::Matrix< T, n, p >& mat );
    void Normalize( Eigen::Matrix< T, n, p >& mat );
    //LogScaleVector();

    Eigen::Matrix< T, n, p > Betas;

    T C = 0.75;
    uint M = 100;
    T rMax;
    T rMin;
    uint statsIt = 1;

};

template< typename T, uint n, uint p >
FOS< T , n, p >::FOS( Eigen::Matrix< T, n, p >& X, Eigen::Matrix< T, n, 1 >& Y ){

}

template< typename T, uint n, uint p >
T FOS< T , n, p >::StdDev( Eigen::Matrix< T, n, p >& mat ) {

    Eigen::RowVectorXf mean = mat.colwise().mean();
    return ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();
}


template< typename T, uint n, uint p >
void FOS< T , n, p >::Normalize( Eigen::Matrix< T, n, p >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}


#endif // FOS_H
