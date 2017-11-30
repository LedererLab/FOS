#!/bin/bash

swig -c++ -python hdim.i

SHARED_ARGS="-std=c++11 -DNDEBUG -DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_EIGEN -fpic -O3 -mtune=native -march=native"
#SWIG_INC_ARGS="-I/usr/include/eigen3 -I/usr/include/viennacl -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lOpenCL"

SWIG_INC_ARGS="-I/usr/include/eigen3 -I/usr/include/viennacl -I/usr/local/cuda/include"
SWIG_LIB_ARGS="-L/usr/local/cuda/lib64 -lOpenCL"

g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../FOS/x_fos.cpp $SWIG_LIB_ARGS -o x_fos.o &

g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../Solvers/SubGradientDescent/ISTA/ista.cpp -o ista.o &
g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../Solvers/SubGradientDescent/FISTA/fista.cpp -o fista.o &
g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../Solvers/CoordinateDescent/coordinate_descent.cpp -o cd.o &

g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../Solvers/SubGradientDescent/ISTA/viennacl_ista.cpp $SWIG_LIB_ARGS -o cl_ista.o &
g++ $SHARED_ARGS $SWIG_INC_ARGS -c ../Solvers/SubGradientDescent/FISTA/viennacl_fista.cpp $SWIG_LIB_ARGS -o cl_fista.o &

g++ $SHARED_ARGS -I/usr/include/python3.5 $SWIG_INC_ARGS -c hdim_wrap.cxx $SWIG_LIB_ARGS & 

wait

g++ -shared x_fos.o fos.o cl_ista.o ista.o cl_fista.o fista.o cd.o hdim_wrap.o $SWIG_LIB_ARGS -o _hdim.so
