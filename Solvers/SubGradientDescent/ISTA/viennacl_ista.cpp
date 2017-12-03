#ifdef W_OPENCL

#include "viennacl_ista.hpp"

template class hdim::CL_ISTA<float,hdim::internal::CL_Solver<float>>;
template class hdim::CL_ISTA<double,hdim::internal::CL_Solver<double>>;

#endif
