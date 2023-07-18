set(LLAMA_STATIC ON CACHE BOOL "")
set(LLAMA_CUBLAS ON CACHE BOOL "")

add_subdirectory("llama.cpp" EXCLUDE_FROM_ALL)
# Without this, the size of struct llama_context_params will be inconsistent
if (LLAMA_CUBLAS)
	target_compile_definitions(llama PUBLIC GGML_USE_CUBLAS)
endif ()
