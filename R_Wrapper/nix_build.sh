#!/bin/bash

# Remove old files
rm -r ./HDIM/src/Generic
rm -r ./HDIM/src/FOS
rm -r ./HDIM/src/Solvers
cp -r ../Screening ./HDIM/src

# Make fresh copy of files from source
cp -r ../Generic ./HDIM/src
cp -r ../FOS ./HDIM/src
cp -r ../Solvers ./HDIM/src
cp -r ../Screening ./HDIM/src

#Let Rcpp update ./HDIM/R/RcppExports.R and ./HDIM/src/RcppExports.R
Rscript ./rcpp_preprocess.R

R CMD INSTALL --preclean --no-multiarch --with-keep.source HDIM
