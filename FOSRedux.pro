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
} else {
    DEFINES += "NDEBUG"
    CONFIG += optimize_full
#    QMAKE_CXXFLAGS *= -Ofast
    QMAKE_CXXFLAGS_RELEASE *= -mtune=native
    QMAKE_CXXFLAGS_RELEASE *= -march=native
}

# Boost
LIBS += -L/usr/local/lib \
        -L/usr/lib \
        -lboost_iostreams \
        -lboost_system \
        -lboost_filesystem \

# Eigen
INCLUDEPATH += /usr/include/eigen3

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

# clBLAS
LIBS += -L/usr/local/lib64/
LIBS += -lclBLAS

# OpenCL
LIBS += -L/usr/local/cuda/lib64
LIBS += -lOpenCL

# Armadillo
LIBS += -larmadillo

HEADERS += FOS/fos.h \
    FOS/test_fos.h \
    Generic/debug.h \
    Generic/generics.h \
    ISTA/ista.h \
    ISTA/test_ista.h \
    R_Wrapper/fos_r.h \
    ISTA/perf_ista.h \
    test_eigen3.h \
    OpenCL_Generics/cl_generics.h \
    OpenCL_Generics/cl_algorithm.h \
    OpenCL_Base/openclbase.h \
    OpenCL_Generics/perf_cl_product.h \
    FOS/test_fos_experimental.h \
    FOS/perf_fos.h \
    FOS/perf_fos_experimental.h \
    FOS/x_fos.h \
    test_armadillo.h \
    OpenCL_Generics/matvectprodtest.h \
    FISTA/fista.h \

SOURCES += main.cpp \
    OpenCL_Base/openclbase.cpp \
    OpenCL_Generics/perf_cl_product.cpp \
    FOS/x_fos.cpp \
    FOS/fos.cpp \
    OpenCL_Generics/matvectprodtest.cpp \

DISTFILES += \
    Python_Wrapper/build.sh \
    Python_Wrapper/eigen.i \
    Python_Wrapper/hdim.i \
    Python_Wrapper/numpy.i
