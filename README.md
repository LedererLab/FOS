# HDIM

HDIM is a toolkit for working with high-dimensional data that emphasizes
speed and statistical guarantees. Specifically, it provides tools for working
with the LASSO objective function.

HDIM provides iterative solvers for the LASSO objective function
 including ISTA, FISTA and Coordinate Descent. HDIM also provides FOS,
  the Fast and Optimal Selection algorithm, a novel new method
for variable selection.

HDIM is a product of the research conducted by
[Lederer and Hauser HDIM Group]( https://lederer.stat.washington.edu/ ).

## Supported Languages

The HDIM package is written in C++, and can be used in native form or via
 Python or R language wrappers.

## Supported Platforms

HDIM currently supports the following operating systems and languages.

| Operating System | Supported Languages |
| ---------------- |:-------------------:|
| Windows          | C++, R              |
| OS X             | C++, Python         |
| Linux            | C++, R, Python      |

# Installation

## C++ ( All Platforms )

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

#### R ( Linux & Windows )

Visit the [HDIM-R](https://github.com/LedererLab/HDIM-R) repository for
installation instructions for the `HDIM` R wrapper.

#### Python ( Linux & OS X )

Visit the [HDIM-Py](https://github.com/LedererLab/HDIM-Py) repository for
installation instructions for the `HDIM` Python wrapper.

## Licensing

The HDIM package is licensed under the MIT license. To
view the MIT license please consult `LICENSE.txt`.

## Authors

* **Benjamin J Phillips** - *Work on native C++, R wrapper, and Python wrapper*
* **Saba Noorassa** - *Work on adding Mac OS compatibility*

## References

[FOS](https://arxiv.org/abs/1609.07195)
