cmake_minimum_required(VERSION 3.12)
project(llmd)

option(LLMD_LLAMA_CPP_STATIC "Whether to build a static library for llama_cpp driver" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if (IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_Release TRUE)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RelWithDebInfo TRUE)
endif ()

add_subdirectory(deps)
add_subdirectory(projects)
