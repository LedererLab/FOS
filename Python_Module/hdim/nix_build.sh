#!/bin/bash

swig -c++ -python hdim.i

SHARED_ARGS="-std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native"
SWIG_OBJ_ARGS="-I/usr/include/eigen3 -c"

g++ $SHARED_ARGS $SWIG_OBJ_ARGS ../src/FOS/x_fos.cpp -o x_fos.o &

g++ $SHARED_ARGS $SWIG_OBJ_ARGS ../src/Solvers/SubGradientDescent/ISTA/ista.cpp -o ista.o &
g++ $SHARED_ARGS $SWIG_OBJ_ARGS ../src/Solvers/SubGradientDescent/FISTA/fista.cpp -o fista.o &
g++ $SHARED_ARGS $SWIG_OBJ_ARGS ../src/Solvers/CoordinateDescent/coordinate_descent.cpp -o cd.o &

g++ $SHARED_ARGS -I/usr/include/python3.5 $SWIG_OBJ_ARGS -c hdim_wrap.cxx &

wait

g++ -shared x_fos.o fos.o ista.o fista.o cd.o hdim_wrap.o -o _hdim.so
