# HDIM

HDIM is a toolkit for working with high-dimensional data that emphasizes
speed and statistical guarantees. Specifically, it provides tools for working
with the LASSO objective function.

HDIM provides traditional iterative solvers for the LASSO objective function
 including ISTA, FISTA and Coordinate Descent. HDIM also provides FOS,
  the Fast and Optimal Selection algorithm, a novel new method
for performing high-dimensional linear regression.

HDIM is a product of the research conducted by
[Lederer and Hauser HDIM Group]( https://lederer.stat.washington.edu/ )
in affiliation with the University of Washington.

## Supported Languages

The HDIM package is written in C++, and can be used in native form or via
the supplied Python or R language wrappers.

## Supported Platforms

HDIM currently supports the following operating systems and languages.

| Operating System | Supported Languages |
| ---------------- |:-------------------:|
| Windows          | C++, R              |
| OS X             | C++, Python         |
| Linux            | C++, R, Python      |

# Installation

## Native C++

HDIM's native code base is cross-platform and header-only.
 There is no installation step -- just clone the git repository and `#include`
 the appropriate source files in your C++ projects. Just be sure to link against the
appropriate dependencies, as outlined in the *Dependencies* section below.

##### Dependencies

HDIM's native code base depends on the following libraries.

* [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)

These libraries will need to be installed in order to use the native C++
 or any of the wrappers.

## Wrappers

Using the Python or R wrapper requires additional dependencies and build steps
compared to using the native code base. Currently both wrappers require building from source --
 in the future we hope to provide users with more convenient installation options.

### Linux

#### Python

##### Dependencies

* [SWIG](http://www.swig.org/download.html) The Simplified Wrapper and Interface Generator
* Python Development Headers
* Numpy, including Development Headers

##### Building

- Clone the HDIM package into a convenient location.
- Navigate to $PKG_DIR/Python_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `nix_build.sh` and mark it as executable ( `chmod +x ./nix_build.sh` ).
- This will run SWIG and build the Python wrapper.
- The Python package path will *not* be updated.

#### R

##### Dependencies

* [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)

##### Building

- Clone the HDIM package into a convenient location.
- Navigate to $PKG_DIR/R_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `nix_build.sh` and mark it as executable ( `chmod +x ./nix_build.sh` ).
- This will run a preprocessing step using Rcpp then build and install the R Wrapper.

### OS X

#### Python

##### Dependencies

* [SWIG](http://www.swig.org/download.html) The Simplified Wrapper and Interface Generator
* Python Development Headers
* Numpy, including Development Headers
* gcc -- installation will not work with the default C++ complier, clang

##### Building

- Clone the HDIM package into a convenient location.
- Navigate to $PKG_DIR/Python_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `os_x_build.sh` and mark it as executable ( `chmod +x ./os_x_build.sh` ).
- This will run SWIG and build the Python wrapper.
- The Python package path will *not* be updated.

### Windows

Installation under Windows *requires* that the root directory of the Eigen3 library
be located in the same directory as the root of the HDIM package.

#### R

##### Dependencies

* [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
* [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)

##### Building

- Clone the HDIM package into the directory where the Eigen3 library is located.
- Navigate to $PKG_DIR/R_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `win_build.ps1` and run it using PowerShell.
- This will run a preprocessing step using Rcpp then build and install the R Wrapper.

## Authors

* **Benjamin J Phillips** - *Work on native C++, R wrapper, and Python wrapper*

## References

[FOS](https://arxiv.org/abs/1609.07195)
