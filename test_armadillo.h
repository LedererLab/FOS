#ifndef TEST_ARMADILLO_H
#define TEST_ARMADILLO_H

#include <iostream>
#include <armadillo>

void arma_mat_vect_prod() {

    arma::mat A = arma::randu< arma::mat >(10000,10000);
    arma::mat B = arma::randu< arma::mat >(10000,1);

//    std::cout << A*B << std::endl;

}

#endif // TEST_ARMADILLO_H
