#!/bin/sh

mkdir -p .build
cd .build

cmake \
	-G "Ninja" \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_TOOLCHAIN_FILE=../cmake/linux.cmake ..
cmake --build .
