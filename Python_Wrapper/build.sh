#!/bin/bash

SPAMS_INCLUDES="-I../../spams/src -I../../spams/src/spams/dictLearn -I../../spams/src/spams/decomp -I../../spams/src/spams/linalg -I../../spams/src/spams/prox"
SPAMS_LINKER="-lstdc++ -lblas -llapack"
SPAMS_OPTS="-fopenmp"

swig -c++ -python hdim.i
#g++ -std=c++11 $SPAMS_OPTS -fpic -I/usr/include/eigen3 $SPAMS_INCLUDES $SPAMS_LINKER -c ../FOS/fos.cpp -o fos.o
g++ -std=c++11 -DNDEBUG -fpic -O3 -I/usr/include/eigen3 -c ../FOS/fos.cpp -o fos.o
g++ -std=c++11 -DNDEBUG -fpic -O0 -I/usr/include/eigen3 -c ../FOS/x_fos.cpp -o x_fos.o
#g++ -std=c++11 -fpic $SPAMS_OPTS $SPAMS_INCLUDES $SPAMS_LINKER -I/usr/include/python2.7 -I/usr/include/eigen3 -c hdim_wrap.cxx
g++ -std=c++11 -DNDEBUG -fpic -O3 -I/usr/include/python3.5 -I/usr/include/eigen3 -c hdim_wrap.cxx
g++ -shared x_fos.o fos.o hdim_wrap.o -o _hdim.so
