#include <iostream>
#include <eigen3/Eigen/Dense>

#include "fos.h"
#include "fosalgorithm.h"

int main(int argc, char *argv[]) {

    Eigen::Matrix< float, 2, 2> m;
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    auto spams_mat = Eigen2SpamsMat< float, 2, 2 >( m );
    std::string print_str;
    spams_mat->print( print_str );

    std::cout << print_str << std::endl;

    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/hdim/App/src/DataSets/riboflavin_t_no_header.csv";

    Eigen::MatrixXd raw_data = CSV2Eigen< Eigen::MatrixXd >( data_set_path );

    auto X = raw_data;
    auto Y = raw_data.col(0);

    FOS< double > algo_fos ( X, Y );
    algo_fos.Algorithm();
}
