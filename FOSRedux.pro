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
#Over-ride Qt's default -O2 optimization
    QMAKE_CXXFLAGS_RELEASE -= -O
    QMAKE_CXXFLAGS_RELEASE -= -O1
    QMAKE_CXXFLAGS_RELEASE -= -O2
    QMAKE_CXXFLAGS_RELEASE *= -O3
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

# SPAMS
INCLUDEPATH +=  ../spams/src \
                ../spams/src/spams/dictLearn \
                ../spams/src/spams/decomp \
                ../spams/src/spams/linalg \
                ../spams/src/spams/prox \

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

# SPAMS has unused parameters in source -- surpress warnings
QMAKE_CXXFLAGS+= -Wno-unused-parameter

# clBLAS
LIBS += -L/usr/local/lib64/
LIBS += -lclBLAS

# OpenCL
LIBS += -L/usr/local/cuda/lib64
LIBS += -lOpenCL

# for FISTA
LIBS += -lstdc++ \
        -lblas \
        -llapack

# Armadillo
LIBS += -larmadillo

HEADERS += FOS/fos.h \
    FOS/test_fos.h \
    Generic/algorithm.h \
    Generic/debug.h \
    Generic/generics.h \
    ISTA/ista.h \
    ISTA/test_ista.h \
    R_Wrapper/fos_r.h \
    SPAMS/test_fista.h \
    ISTA/perf_ista.h \
    SPAMS/perf_fista.h \
    test_eigen3.h \
    OpenCL_Generics/cl_generics.h \
    OpenCL_Generics/cl_algorithm.h \
    OpenCL_Base/openclbase.h \
    OpenCL_Generics/perf_cl_product.h \
    FOS/test_fos_experimental.h \
    FOS/perf_fos.h \
    FOS/perf_fos_experimental.h \
    FOS/x_fos.h \
#    FOS/fos_imperative.h \
    test_armadillo.h \
    OpenCL_Generics/matvectprodtest.h

SOURCES += main.cpp \
    OpenCL_Base/openclbase.cpp \
    OpenCL_Generics/perf_cl_product.cpp \
    FOS/x_fos.cpp \
    FOS/fos.cpp \
    OpenCL_Generics/matvectprodtest.cpp

DISTFILES += \
    Python_Wrapper/build.sh \
    Python_Wrapper/eigen.i \
    Python_Wrapper/hdim.i \
    Python_Wrapper/numpy.i
