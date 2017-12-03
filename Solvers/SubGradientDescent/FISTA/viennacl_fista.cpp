//Explicit Instantiation of VCL FISTA for Python Wrappers
#ifdef W_OPENCL
#include "viennacl_fista.hpp"

template class hdim::CL_FISTA<float,hdim::internal::CL_Solver<float>>;
template class hdim::CL_FISTA<double,hdim::internal::CL_Solver<double>>;

#endif
