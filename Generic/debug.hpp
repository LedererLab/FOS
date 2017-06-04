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
#include <cxxabi.h>// abi::__cxa_demangle
// Boost Headers
//
// Miscellaneous Headers
//

/*! \file
 *  \brief Functions designed to aid in debugging.
 */

template <typename T>
/*!
 * \brief Get the de-mangled name of a type ( as it would
 * appear in the source code ).
 *
 * \return
 * name of the template parameter type
 */
std::string get_type_name () {
    int status;
    char* type_name = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
    std::string type_str = std::string( type_name );

    free (type_name);

    return type_str;
}

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
