#include <iostream>
#include <eigen3/Eigen/Dense>

#include "fos.h"
#include "fosalgorithm.h"
#include "tests.h"

int main(int argc, char *argv[]) {

    std::string data_set_path = "/home/bephillips2/Desktop/Hanger Bay 1/Academia/HDIM/hdim/App/src/DataSets/riboflavin_t_no_header.csv";

    Eigen::MatrixXd raw_data = CSV2Eigen< Eigen::MatrixXd >( data_set_path );

    auto X = raw_data;
    auto Y = raw_data.col(0);

    FOS< double > algo_fos ( X, Y );
    algo_fos.Algorithm();

}
