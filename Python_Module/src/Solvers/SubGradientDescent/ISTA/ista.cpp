//Explicit Instantiation of ISTA for Python Wrappers

#include "ista.hpp"

template class hdim::ISTA<float,hdim::internal::Solver<float>>;
template class hdim::ISTA<float,hdim::internal::ScreeningSolver<float>>;

template class hdim::ISTA<double,hdim::internal::Solver<double>>;
template class hdim::ISTA<double,hdim::internal::ScreeningSolver<double>>;
