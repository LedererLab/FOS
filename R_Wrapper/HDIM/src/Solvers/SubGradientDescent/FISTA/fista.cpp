//Explicit Instantiation of FISTA for Python Wrappers

#include "fista.hpp"

template class hdim::FISTA<float,hdim::internal::Solver<float>>;
template class hdim::FISTA<float,hdim::internal::ScreeningSolver<float>>;

template class hdim::FISTA<double,hdim::internal::Solver<double>>;
template class hdim::FISTA<double,hdim::internal::ScreeningSolver<double>>;
