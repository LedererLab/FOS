//Explicit Instantiation of X_FOS for Wrappers


#include "fos_js.hpp"

template class JS_FOS<float>;
template class JS_FOS<double>;

#ifdef JS_BUILD

#include <emscripten/bind.h>
using namespace emscripten;


EMSCRIPTEN_BINDINGS(stl_wrappers) {
    register_vector<float>("VectorFloat");
    register_vector<double>("VectorDouble");
    register_vector<int>("VectorInt");
}

// Binding code
EMSCRIPTEN_BINDINGS(fos_interface) {
    class_< JS_FOS<double> >("FOS")
    .constructor()
    .function( "Run", &JS_FOS<double>::operator() )
    .function( "ReturnLambda", &JS_FOS<double>::ReturnLambda )
    .function( "ReturnIntercept", &JS_FOS<double>::ReturnIntercept )
    .function( "ReturnOptimIndex", &JS_FOS<double>::ReturnOptimIndex )
    .function( "ReturnCoefficients", &JS_FOS<double>::ReturnCoefficients )
    .function( "ReturnSupport", &JS_FOS<double>::ReturnSupport )
    ;
}

#endif
