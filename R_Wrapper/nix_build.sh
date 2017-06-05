#!/bin/bash

rm -r ./HDIM/src/Generic
rm -r ./HDIM/src/FOS
rm -r ./HDIM/src/Solvers

cp -r ../Generic ./HDIM/src
cp -r ../FOS ./HDIM/src
cp -r ../Solvers ./HDIM/src

#Let Rcpp update ./HDIM/R/RcppExports.R and ./HDIM/src/RcppExports.R
Rscript ./rcpp_preprocess.R

R CMD INSTALL --preclean --no-multiarch --with-keep.source HDIM
