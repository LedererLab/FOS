#!/bin/bash

SHARED_ARGS="-std=c++11 -DJS_BUILD -DNDEBUG -O2"
SWIG_OBJ_ARGS="-I$HOME/Packages"

EMCC_CMD="/home/bephillips2/Packages/emsdk-portable/emscripten/1.37.16/emcc"

$EMCC_CMD $SHARED_ARGS $SWIG_OBJ_ARGS --bind -o fos.js fos_js.cpp

# /home/bephillips2/Packages/emsdk-portable/emscripten/1.37.16/emcc -std=c++11 -DNDEBUG -DJS_BUILD -O2 -I$HOME/Packages --bind -o fos.js ../FOS/x_fos.cpp
