#!/bin/bash

cp -r ../Generic ./HDIM/src
cp -r ../FOS ./HDIM/src
cp -r ../Solvers ./HDIM/src

R CMD INSTALL --preclean --no-multiarch --with-keep.source HDIM
