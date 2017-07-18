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

#gnuplot iostreams
INCLUDEPATH += ../gnuplot-iostream

# Boost
LIBS += -L/usr/local/lib \
        -L/usr/lib \
        -lboost_iostreams \
        -lboost_system \
        -lboost_filesystem \

# Eigen
INCLUDEPATH += /usr/include/eigen3

### clBLAS
#LIBS += -L/usr/local/lib64/
#LIBS += -lclBLAS

### OpenCL
#LIBS += -L/usr/local/cuda/lib64
#LIBS += -lOpenCL

HEADERS += FOS/fos.hpp \
    Generic/debug.hpp \
    Generic/generics.hpp \
    Solvers/SubGradientDescent/ISTA/ista.hpp \
    Solvers/SubGradientDescent/ISTA/test_ista.hpp \
    R_Wrapper/fos_r.hpp \
    FOS/x_fos.hpp \
    Solvers/SubGradientDescent/FISTA/fista.hpp \
    Solvers/CoordinateDescent/coordinate_descent.hpp \
    Solvers/SubGradientDescent/subgradient_descent.hpp \
    Solvers/solver.hpp \
    FOS/duality.hpp \
    Screening/screening_rules.hpp \
    Solvers/CoordinateDescent/coordinatedescentwithscreen.hpp \
    FOS/test_x_fos.hpp

SOURCES += main.cpp \
    FOS/x_fos.cpp \
    FOS/fos.cpp \
    Solvers/SubGradientDescent/ISTA/ista.cpp \
    Solvers/SubGradientDescent/FISTA/fista.cpp \
    Solvers/CoordinateDescent/coordinate_descent.cpp \
    Solvers/CoordinateDescent/coordinatedescentwithscreen.cpp

DISTFILES += \
    Python_Wrapper/build.sh \
    Python_Wrapper/eigen.i \
    Python_Wrapper/hdim.i \
    Python_Wrapper/numpy.i
