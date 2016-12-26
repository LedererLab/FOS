#include <iostream>
#include <eigen3/Eigen/Dense>

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
    auto fista_d = FistaFlat< float, 2,  2 >( spams_mat, spams_mat, spams_mat, 1.0f );
    fista_d->print(print_str);

    std::cout << print_str << std::endl;

}
