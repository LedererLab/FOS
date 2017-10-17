#!/bin/bash

swig -c++ -python hdim.i

# Set default compiler to `g++` if not defined by hand. You might have to install traditional `gcc` through
# something like `brew` because I believe Mac OS calls `clang` when `g++` is called by default. On
# my machine this script works with:
#
# CXX=g++-4.9 ./os_x_build.sh

CXX=${CXX-g++}
SHARED_ARGS="-std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native"

$CXX $SHARED_ARGS -I/usr/include/eigen3 -c ../FOS/fos.cpp -o fos.o
$CXX $SHARED_ARGS -I/usr/include/eigen3 -c ../FOS/x_fos.cpp -o x_fos.o

$CXX $SHARED_ARGS -I/usr/include/eigen3 -c ../Solvers/SubGradientDescent/ISTA/ista.cpp -o ista.o
$CXX $SHARED_ARGS -I/usr/include/eigen3 -c ../Solvers/SubGradientDescent/FISTA/fista.cpp -o fista.o
$CXX $SHARED_ARGS -I/usr/include/eigen3 -c ../Solvers/CoordinateDescent/coordinate_descent.cpp -o cd.o

$CXX $SHARED_ARGS -I/usr/include/eigen3                                                \
  -I /usr/local/lib/python3.6/site-packages/numpy/core/include                         \
  -I /usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6/Headers  \
  -c hdim_wrap.cxx

$CXX -dynamiclib -shared x_fos.o fos.o ista.o fista.o cd.o hdim_wrap.o                 \
  -o _hdim.so `python3-config --ldflags` `python3-config --cflags`
