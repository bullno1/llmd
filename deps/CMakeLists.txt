set(LLAMA_STATIC ON CACHE BOOL "")
set(LLAMA_LTO ON CACHE BOOL "")
add_subdirectory("llama.cpp" EXCLUDE_FROM_ALL)
