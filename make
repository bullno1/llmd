#!/bin/sh

mkdir -p .build
cd .build

cmake -G "Ninja Multi-Config" -DCMAKE_TOOLCHAIN_FILE=../cmake/linux.cmake ..
cmake --build . --config RelWithDebInfo
