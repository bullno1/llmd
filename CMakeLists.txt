cmake_minimum_required(VERSION 3.12)

project(llmd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLMD_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(deps)
add_subdirectory(src)
