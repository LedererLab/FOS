TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#Force use of c++14
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS -= -std=c++0x

#CONFIG += c++11

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
INCLUDEPATH +=  ../spams/src \
                ../spams/src/spams/dictLearn \
                ../spams/src/spams/decomp \
                ../spams/src/spams/linalg \
                ../spams/src/spams/prox \

QMAKE_CXXFLAGS+= -fopenmp
#QMAKE_CXXFLAGS+= -O3
QMAKE_LFLAGS +=  -fopenmp

QMAKE_CXXFLAGS+= -Wno-unused-parameter

#OpenCL
#LIBS +=-L "/usr/local/cuda/lib64" -lOpenCL

#???
LIBS += -lstdc++ \
        -lblas \
        -llapack

HEADERS += fos.h \
    fosalgorithm.h

SOURCES += main.cpp \
    #fos.tpp
