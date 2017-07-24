//Explicit Instantiation of CD for Python Wrappers

#include "coordinate_descent.hpp"

template class hdim::LazyCoordinateDescent<float,hdim::internal::Solver<float>>;
template class hdim::LazyCoordinateDescent<float,hdim::internal::ScreeningSolver<float>>;

template class hdim::LazyCoordinateDescent<double,hdim::internal::Solver<double>>;
template class hdim::LazyCoordinateDescent<double,hdim::internal::ScreeningSolver<double>>;
