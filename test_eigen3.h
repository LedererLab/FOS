#ifndef TESTS_H
#define TESTS_H

// C System-Headers
//
// C++ System headers
//
// Eigen Headers
#include <eigen3/Eigen/Dense>
// Boost Headers
//
// SPAMS Headers
//
// Armadillo Headers
//
// Project Specific Headers
#include "fos_debug.h"

void crossprod_test() {

    DEBUG_PRINT( __func__ );
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > X( 2 ,2 );
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > Y( 2, 1 );

    X << 1.0f, 2.0f, 3.0f, 4.0f;
    Y << 1.0f, 3.0f;

    auto cross_prod =  X.transpose()*Y;

    DEBUG_PRINT( "X:\n" << X << "\nY:\n" << Y );
    DEBUG_PRINT( "Cross Product X and Y: \n" << cross_prod );
    DEBUG_PRINT( "Answer should be: \n 10 \n 14" );

}

void csv_read_test() {

    DEBUG_PRINT( __func__ );
    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/test_data.csv";
    Eigen::MatrixXd raw_data = CSV2Eigen< Eigen::MatrixXd >( data_set_path );

    DEBUG_PRINT( "Matrix as read from file:\n" << raw_data );
    DEBUG_PRINT( "Answer should be: \n 1, 2 \n 3, 4" );

}


void lp_norm_test() {

    DEBUG_PRINT( __func__ );
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > X( 2 ,2 );

    X << 1.0f, 2.0f, 3.0f, 4.0f;

    DEBUG_PRINT( "X:\n" << X );

    DEBUG_PRINT( "L1 Norm of X: " << X.template lpNorm< 1 >() );
    DEBUG_PRINT( "Should be: " << 10 );

    DEBUG_PRINT( "L2 Norm Squared of X: " << X.squaredNorm() );
    DEBUG_PRINT( "Should be: " << 30 );

    DEBUG_PRINT( "L-infinity Norm of X: " << X.template lpNorm< Eigen::Infinity >() );
    DEBUG_PRINT( "Should be: " << 4 );

}

void min_max_test() {

    DEBUG_PRINT( __func__ );
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > X( 2 ,2 );

    X << 1.0f, -2.0f, 3.0f, -4.0f;

    double max_elem = static_cast< float >( X.cwiseAbs().maxCoeff() );

    DEBUG_PRINT( "X:\n" << X );
    DEBUG_PRINT( "Absolute Maximum value in X: " << max_elem );
    DEBUG_PRINT( "Should be: " << 4 );

}

void row_col_test() {

    DEBUG_PRINT( __func__ );
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > X( 2 ,2 );
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > Y( 1, 2 );

    X << 1.0f, 2.0f, 3.0f, 4.0f;
    Y << 1.0f, 3.0f;

    DEBUG_PRINT( "X:\n" << X << "Y:\n" << Y );
    DEBUG_PRINT( "First Column of X:\n " << X.col(0) );
    DEBUG_PRINT( "First Column of Y:\n " << Y.col(0) );

}

void RunTests() {

    crossprod_test();
    csv_read_test();
    lp_norm_test();
    min_max_test();
    row_col_test();
}

#endif // TESTS_H
