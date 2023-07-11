#!/bin/sh

mkdir -p build
cd build

export CC=$(which clang)
export CXX=$(which clang++)
cmake -G "Ninja Multi-Config" -DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON ..
mold -run cmake --build . --config RelWithDebInfo
