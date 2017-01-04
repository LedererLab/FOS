#ifndef FOS_DEBUG_H
#define FOS_DEBUG_H

/*! \file
 *  \brief Preprocessor macros used for debugging and profiling
 */

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
# define DEBUG_PRINT(x) do {} while (0)
#endif

#endif // FOS_DEBUG_H
