cmake_minimum_required(VERSION 3.12)
project(llmd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if (IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_Release TRUE)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RelWithDebInfo TRUE)
endif ()

# Add deps first
add_subdirectory(deps)

# Then project options here to not leak them to dependencies

option(LLMD_LLAMA_CPP_STATIC "Whether to build a static library for llama_cpp driver" OFF)
option(LLMD_IPC_CLIENT_STATIC "Whether to build a static library for ipc client" OFF)
option(LLMD_IPC_SERVER_STATIC "Whether to build a static library for ipc server" ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

function (setup_library LIBRARY_NAME IS_STATIC SOURCES)
	if (IS_STATIC)
		add_library(${LIBRARY_NAME} STATIC ${SOURCES})
	else ()
		string(TOUPPER ${LIBRARY_NAME} LIBRARY_NAME_UPPER)
		add_library(${LIBRARY_NAME} SHARED ${SOURCES})
		target_compile_definitions(${LIBRARY_NAME} PUBLIC ${LIBRARY_NAME_UPPER}_SHARED)
		target_compile_definitions(${LIBRARY_NAME} PRIVATE ${LIBRARY_NAME_UPPER}_BUILD)
		set_target_properties(${LIBRARY_NAME} PROPERTIES C_VISIBILITY_PRESET hidden)
	endif ()

	if (NOT MSVC)
		target_link_options(${LIBRARY_NAME} PRIVATE
			-Wl,--exclude-libs,ALL
			-Wl,--no-whole-archive
		)
	endif ()
endfunction ()

add_subdirectory(libs)
add_subdirectory(examples)
