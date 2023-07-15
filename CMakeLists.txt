cmake_minimum_required(VERSION 3.12)
project(llmd)

option(LLMD_LLAMA_CPP_STATIC "Whether to build a static library for llama_cpp driver" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if (IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_Release TRUE)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RelWithDebInfo TRUE)
endif ()

add_subdirectory(deps)
add_subdirectory(projects)
