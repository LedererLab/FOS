//Explicit Instantiation of VCL FISTA for Python Wrappers

#include "viennacl_fista.hpp"

template class hdim::vcl::FISTA<float,hdim::vcl::internal::Solver<float>>;
template class hdim::vcl::FISTA<double,hdim::vcl::internal::Solver<double>>;
