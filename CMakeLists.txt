cmake_minimum_required(VERSION 3.12)
project(llmd)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

if (IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

option(LLMD_LLAMA_CPP_STATIC "Whether to build a static library instead" OFF)

add_subdirectory(deps)
add_subdirectory(src)
