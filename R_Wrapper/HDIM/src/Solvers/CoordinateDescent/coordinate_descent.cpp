//Explicit Instantiation of CD for Python Wrappers

#include "coordinate_descent.hpp"

template class hdim::CoordinateDescentSolver<float>;
template class hdim::CoordinateDescentSolver<double>;
