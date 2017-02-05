#ifndef DEBUG_H
#define DEBUG_H

#ifdef DEBUG
#define DEBUG_ON 1
#else
#define DEBUG_ON 0
#endif

// C System-Headers
#include <stdio.h>
// C++ System headers
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>
// Boost Headers
//
// Miscellaneous Headers
//

#define TIME_IT( func, ... ) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        func \
        auto end = std::chrono::high_resolution_clock::now();\
        std::chrono::duration<double, std::milli> ms = end - start;\
        auto time_taken = ms.count();\
        std::cout<< "Function took " << time_taken <<" ms." << std::endl;\
    } while (0)



#ifdef DEBUG
# define DEBUG_PRINT( x, ... ) std::cout << x __VA_ARGS__ << std::endl;
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#endif // DEBUG_H
