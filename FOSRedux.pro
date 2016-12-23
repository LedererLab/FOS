TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#Force use of c++14
QMAKE_CXXFLAGS += -std=c++14
QMAKE_CXXFLAGS -= -std=c++0x

CONFIG(debug, debug|release) {
    DEFINES += "DEBUG"
} else {
    DEFINES += "NDEBUG"
    QMAKE_CXXFLAGS -= -O2
    QMAKE_CXXFLAGS += -O3
}

#Eigen BLAS
INCLUDEPATH += /usr/include/eigen3

#FISTA
INCLUDEPATH += ../spams/src \
                ../spams/src/spams

#OpenCL
LIBS +=-L "/usr/local/cuda/lib64" -lOpenCL

SOURCES += main.cpp \
    fos.cpp \
    fos.tpp

HEADERS += \
    fos.h \
    fosalgorithm.h
