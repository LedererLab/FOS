TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#Force use of c++11
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS -= -std=gnu++11
QMAKE_CXXFLAGS -= -std=c++0x

CONFIG += c++11

# Pass flags for bebugging
# Override Qt's default -O2 flag in release mode
CONFIG(debug, debug|release) {
    DEFINES += "DEBUG"
    DEFINES += "VIENNACL_WITH_OPENCL"
    DEFINES += "VIENNACL_WITH_EIGEN"
} else {
    DEFINES += "NDEBUG"
    DEFINES += "VIENNACL_WITH_OPENCL"
    DEFINES += "VIENNACL_WITH_EIGEN"
    CONFIG += optimize_full
    QMAKE_CXXFLAGS_RELEASE *= -mtune=native
    QMAKE_CXXFLAGS_RELEASE *= -march=native
}

# Boost
LIBS += -L/usr/local/lib \
        -L/usr/lib \
        -lboost_iostreams \
        -lboost_system \
        -lboost_filesystem \

# ViennaCL
INCLUDEPATH += /usr/include/viennacl

# OpenCL
INCLUDEPATH +=-I "/usr/local/cuda/include"
LIBS +=-L "/usr/local/cuda/lib64" -lOpenCL

# Eigen
INCLUDEPATH += /usr/include/eigen3

# emscripten
INCLUDEPATH += /home/bephillips2/Packages/emsdk-portable/emscripten/1.37.16/system/include/

HEADERS += Generic/debug.hpp \
    Generic/generics.hpp \
    Solvers/SubGradientDescent/ISTA/ista.hpp \
    Solvers/SubGradientDescent/ISTA/test_ista.hpp \
    R_Wrapper/fos_r.hpp \
    FOS/x_fos.hpp \
    Solvers/SubGradientDescent/FISTA/fista.hpp \
    Solvers/CoordinateDescent/coordinate_descent.hpp \
    Solvers/SubGradientDescent/subgradient_descent.hpp \
    Solvers/solver.hpp \
    Screening/screening_rules.hpp \
    Solvers/CoordinateDescent/coordinatedescentwithscreen.hpp \
    FOS/test_x_fos.hpp \
    Solvers/screeningsolver.hpp \
    Solvers/abstractsolver.hpp \
    FOS/perf_fos.hpp \
    Solvers/SubGradientDescent/ISTA/perf_ista.hpp \
    JS_Wrapper/fos_js.hpp \
    OpenCL_Base/openclbase.h \
    OpenCL_Generics/cl_algorithm.h \
    OpenCL_Generics/cl_generics.h \
    Generic/ocl_debug.hpp \
    Solvers/SubGradientDescent/viennacl_subgradient_descent.hpp \
    Solvers/viennacl_solver.hpp \
    Solvers/viennacl_abstractsolver.hpp \
    Solvers/SubGradientDescent/ISTA/viennacl_ista.hpp \
    Solvers/base_solver.hpp \
    Solvers/SubGradientDescent/FISTA/viennacl_fista.hpp \
    Solvers/SubGradientDescent/FISTA/test_fista.hpp \
    Solvers/SubGradientDescent/FISTA/perf_fista.hpp

SOURCES += main.cpp \
    FOS/x_fos.cpp \
    Solvers/SubGradientDescent/ISTA/ista.cpp \
    Solvers/SubGradientDescent/FISTA/fista.cpp \
    Solvers/CoordinateDescent/coordinate_descent.cpp \
    Solvers/CoordinateDescent/coordinatedescentwithscreen.cpp \
    JS_Wrapper/fos_js.cpp \
    OpenCL_Base/openclbase.cpp \
    Generic/ocl_debug.cpp \
    Solvers/SubGradientDescent/ISTA/viennacl_ista.cpp \
    Solvers/SubGradientDescent/FISTA/viennacl_fista.cpp

DISTFILES += \
    Python_Wrapper/build.sh \
    Python_Wrapper/eigen.i \
    Python_Wrapper/hdim.i \
    Python_Wrapper/numpy.i
