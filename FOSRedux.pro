TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#Force use of c++14
QMAKE_CXXFLAGS += -std=c++14
QMAKE_CXXFLAGS -= -std=gnu++11
QMAKE_CXXFLAGS -= -std=c++0x

CONFIG += c++14

# Pass flags for bebugging
# Override Qt's default -O2 flag in release mode
CONFIG(debug, debug|release) {
    DEFINES += "DEBUG"
} else {
    DEFINES += "NDEBUG"
    QMAKE_CXXFLAGS -= -O2
    QMAKE_CXXFLAGS += -O3
}

# Rcpp
INCLUDEPATH += /usr/local/lib/R/site-library/Rcpp/include/

# Eigen
INCLUDEPATH += /usr/include/eigen3

# SPAMS
INCLUDEPATH +=  ../spams/src \
                ../spams/src/spams/dictLearn \
                ../spams/src/spams/decomp \
                ../spams/src/spams/linalg \
                ../spams/src/spams/prox \

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS +=  -fopenmp

# SPAMS has unused parameters in source -- surpress warnings
QMAKE_CXXFLAGS+= -Wno-unused-parameter

#OpenCL
#LIBS +=-L "/usr/local/cuda/lib64" -lOpenCL

#for FISTA
LIBS += -lstdc++ \
        -lblas \
        -llapack

#Armadillo
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
    SPAMS/perf_fista.h

SOURCES += main.cpp \
    #FOS/fos.tpp \
    R_Wrapper/fos_r.cpp
