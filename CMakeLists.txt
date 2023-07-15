cmake_minimum_required(VERSION 3.12)
project(llmd)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

option(LLMD_LLAMA_CPP_STATIC "Whether to build a static library for llama_cpp driver" OFF)

if (IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_Release TRUE)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RelWithDebInfo TRUE)
endif ()

add_subdirectory(deps)
add_subdirectory(projects)
