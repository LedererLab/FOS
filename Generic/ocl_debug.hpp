#ifndef OCL_DEBUG_HPP
#define OCL_DEBUG_HPP

/*! \file
 *  \brief Preprocessor macros used for debugging OpenCL specific sections of code
 */

// C System-Headers
#include <stdio.h>
// C++ System headers
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <assert.h> // static_assert
#include <fstream>
#include <typeinfo>  // typeid, typeof


#if defined __linux__ || defined __APPLE__
#include <cxxabi.h>// abi::__cxa_demangle
#endif

// Boost Headers
#include <boost/lexical_cast.hpp>
// OpenCL Headers
#include <CL/cl.h>
#include <CL/cl.hpp>

std::string CLErrorToString(cl_int error);
std::string CLErrorToString(cl_int* error);

#ifdef DEBUG
#define OCL_DEBUG(err, ...) \
            do { std::cout << __func__ <<\
        __FILE__ <<\
        " Line# "<<\
        __LINE__ <<\
        " OpenCL Status: " <<\
        CLErrorToString( err ) << std::endl; } while (0)
#else
#define OCL_DEBUG( err, ... )
#endif

#endif // OCL_DEBUG_HPP
