#ifndef FOS_DEBUG_H
#define FOS_DEBUG_H

/*! \file
 *  \brief Preprocessor macros used for debugging and profiling
 */

// C System-Headers
#include <stdio.h>
// C++ System headers
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <assert.h> //static_assert
#include <fstream>
#include <typeinfo>  // typeid, typeof

#if defined __linux__ || defined __APPLE__
#include <cxxabi.h>// abi::__cxa_demangle
#endif

// Boost Headers
//
// Miscellaneous Headers
//

/*! \file
 *  \brief Functions designed to aid in debugging.
 */

/*!
 * \brief Get the de-mangled name of a type ( as it would
 * appear in the source code ).
 *
 * \return
 * name of the template parameter type
 */
#if defined __linux__ || defined __APPLE__
template <typename T>
std::string get_type_name () {

    int status;
    char* type_name = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
    std::string type_str = std::string( type_name );

    free (type_name);

    return type_str;
}
#elif _WIN32
std::string get_type_name() {
    return std::string( typeid(T).name() );
}
#endif

/** Measure how long a function takes to execute.*/
#define TIME_IT( func, ... ) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        func \
        auto end = std::chrono::high_resolution_clock::now();\
        std::chrono::duration<double, std::milli> ms = end - start;\
        auto time_taken = ms.count();\
        std::cout<< "Function took " << time_taken <<" ms." << std::endl;\
    } while (0)

/** Print a sequence of statements if program is compilied with the -DDEBUG flag. */
#ifdef DEBUG
# define DEBUG_PRINT( x, ... ) std::cout << x __VA_ARGS__ << std::endl;
#else
# define DEBUG_PRINT( x, ... )
#endif

#endif // FOS_DEBUG_H
