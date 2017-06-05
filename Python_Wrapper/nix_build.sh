#!/bin/bash

swig -c++ -python hdim.i

g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/eigen3 -c ../FOS/fos.cpp -o fos.o
g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/eigen3 -c ../FOS/x_fos.cpp -o x_fos.o

g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/eigen3 -c ../Solvers/SubGradientDescent/ISTA/ista.cpp -o ista.o
g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/eigen3 -c ../Solvers/SubGradientDescent/FISTA/fista.cpp -o fista.o
g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/eigen3 -c ../Solvers/CoordinateDescent/coordinate_descent.cpp -o cd.o

g++ -std=c++11 -DNDEBUG -fpic -O3 -mtune=native -march=native -I/usr/include/python3.5 -I/usr/include/eigen3 -c hdim_wrap.cxx
g++ -shared x_fos.o fos.o ista.o fista.o cd.o hdim_wrap.o -o _hdim.so
