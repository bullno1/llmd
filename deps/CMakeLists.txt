set(LLAMA_STATIC ON CACHE BOOL "")
set(LLAMA_CUBLAS ON CACHE BOOL "")

if (NOT LLMD_LLAMA_CPP_STATIC AND NOT MSVC)
	add_compile_options(-fPIC)
endif ()

add_subdirectory("llama.cpp" EXCLUDE_FROM_ALL)
